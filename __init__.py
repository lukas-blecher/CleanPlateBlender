
import os, sys
paths = [
    r'PYTHON_PATH/site-packages',
    os.path.dirname(os.path.realpath(__file__)),
    r'C:\Users\user\Desktop\ML\CleanPlateBlender'
]

for p in paths:
    sys.path.insert(0, p)
from utils import io
from tools import infer_liteflownet, frame_inpaint, propagation_inpaint, interpolate
from models import LiteFlowNet, resnet_models
from dataset import FlowInfer, data_list, FlowInitial
import numpy as np
from PIL import Image, ImageDraw
import cvbase as cvb
from geomdl import BSpline, utilities
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import cv2
import bpy
from bpy.types import Operator, Panel, PropertyGroup, WindowManager
from bpy.props import PointerProperty, StringProperty, IntProperty, FloatProperty, BoolProperty, EnumProperty
import tempfile


bl_info = {
    'blender': (2, 80, 0),
    'name': 'CleanPlate',
    'category': 'Motion Tracking',
    'location': 'Masking > Movie Clip Editor > CleanPlate',
    'author': 'Lukas Blecher'
}


class Arguments:
    LiteFlowNet, DFC, ResNet101 = True, True, True
    MASK_MODE = None
    INITIAL_HOLE = True
    get_mask = True


def mask_name_callback(scene, context):
    items = []
    masks = bpy.data.masks
    for i, m in enumerate(masks):
        mname = m.name
        items.append((mname, mname, '', 'MOD_MASK', i+1))
    return items


class Settings(PropertyGroup):

    mask_name: EnumProperty(
        name="Mask",
        description="Which Mask to use for the inpainting",
        items=mask_name_callback,
        options=set()
    )

    mask_enlarge: IntProperty(
        name="Enlarge Mask",
        description="Enlarge the effective mask during inpainting",
        default=0,
        min=0,
        options=set()
    )

    n_threads: IntProperty(
        name="Threads",
        description="Number of threads",
        default=0,
        min=0,
        options=set()
    )

    th_warp: IntProperty(
        name="Threshold",
        description="Threshold in the propagation process",
        default=40,
        min=1,
        options=set()
    )

    batch_size: IntProperty(
        name="Batch size",
        description="How many frames to process at once\n (depends heavily on the GPU)",
        default=1,
        min=1,
        options=set()
    )

    outpath: StringProperty(
        name="Output Directory",
        description="Where to save the inpainted images",
        default=tempfile.gettempdir(),
        maxlen=1024,
        subtype='DIR_PATH',
        options=set()
    )

    imgending: StringProperty(
        name="File Format",
        description="File Format for the inpainted images",
        default='png',
        options=set()
    )

    downscale: FloatProperty(
        name="Downscaling Factor",
        description="How much to downscale the image",
        default=1,
        min=1,
        options=set()
    )

    steps: IntProperty(
        name="Interpolation steps",
        description="How many steps in between 2 frames to generate",
        default=1,
        min=0,
        options=set()
    )


def spline2mask(crl, width, height, delta=.05, new_shape=None):
    c, r, l = crl if type(crl) == list else crl.tolist()
    cps = []
    for i in range(len(c)):
        ip = (i+1) % len(c)
        cps.append([c[i], r[i], l[ip], c[ip]])
    connecs = []
    for i in range(len(cps)):
        curve = BSpline.Curve()
        curve.degree = 3
        curve.ctrlpts = cps[i]
        curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
        # print('delta',delta)
        curve.delta = delta
        curve.evaluate()
        connecs.append(curve.evalpts)

    polygon = np.array(connecs).flatten().tolist()
    img = Image.new('L', (height, width), 255)
    ImageDraw.Draw(img).polygon(polygon, outline=0, fill=0)
    if new_shape is None:
        new_shape = (height, width)
    mask = np.array(img.resize(new_shape, Image.NEAREST))
    #print(mask.shape, width, height, downscale,int(width//downscale), int(height//downscale))
    return mask == False


class CleanPlateMaker:
    state = -1
    complete = False

    def big_trans(self, inv=False):
        return lambda x: x

    def small_trans(self, inv=False):
        frac = max(self.hw)/float(min(self.hw))
        off = .5-1/(2.*frac)
        if not inv:
            return lambda x: (x-off)*frac
        else:
            return lambda x: (x/frac)+off

    def copy_point_attributes(self, point, new_point):
        attributes = ['co', 'handle_left', 'handle_left_type', 'handle_right', 'handle_right_type', 'handle_type', 'weight']
        for attr in attributes:
            setattr(new_point, attr, getattr(point, attr))

    def absolute_coord(self, coordinate):
        width, height = self.hw
        coord = coordinate.copy()
        return [self.xtrans(coord.x)*width, (1-self.ytrans(coord.y))*height]

    def relative_coord(self, coordinate):
        width, height = self.hw
        return [self.xinvt(coordinate[0]/float(width)), self.yinvt(1-(coordinate[1]/float(height)))]

    def set_coordinate_transform(self):
        if self.hw[0] < self.hw[1]:
            self.xtrans = self.small_trans()
            self.xinvt = self.small_trans(True)
            self.ytrans = self.big_trans()
            self.yinvt = self.big_trans()
        elif self.hw[0] > self.hw[1]:
            self.ytrans = self.small_trans()
            self.yinvt = self.small_trans(True)
            self.xtrans = self.big_trans()
            self.xinvt = self.big_trans()
        else:
            self.xtrans = self.big_trans()
            self.xinvt = self.big_trans()
            self.ytrans = self.big_trans()
            self.yinvt = self.big_trans()

    def collect_next_frame(self):
        self.i += 1
        ret, frame = self.cap.read()
        curr_frame = bpy.context.scene.frame_current
        # state finished. return and go to next state
        if not ret or curr_frame == self.frame_end+1:
            self.cap.release()
            bpy.ops.clip.change_frame(frame=self.frame_end)
            self.i = -1
            self.state += 1
            return
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(self.args.img_root, '%05d.%s' % (curr_frame, self.settings.imgending)), frame)
        #self.frames[self.i] = np.array(frame)/255
        raw_mask = np.zeros((self.H, self.W), dtype=np.uint8)
        for layer in self.mask.layers:
            if layer.hide_render:
                continue
            maskSplines = layer.splines
            for _, maskSpline in enumerate(maskSplines):
                points = maskSpline.points
                maskSpline.use_cyclic = True
                co, lhand, rhand = [], [], []
                for p in points:
                    # need types to be free as it is the most general type
                    p.handle_left_type = 'FREE'
                    p.handle_right_type = 'FREE'
                    co.append(self.absolute_coord(p.co))
                    lhand.append(self.absolute_coord(p.handle_left))
                    rhand.append(self.absolute_coord(p.handle_right))
                # collection of coordinates and handles
                crl = [co, rhand, lhand]
                # get mask from the point coordinates
                raw_mask += spline2mask(crl, self.hw[1], self.hw[0], new_shape=(self.W, self.H)).astype(np.uint8)
        raw_mask = np.clip(raw_mask, 0, 1)

        Image.fromarray(raw_mask*255).resize((self.W, self.H), Image.BILINEAR).save(os.path.join(self.args.mask_root, '%05d.png' % curr_frame))
        #canvas.save(os.path.join(self.settings.outpath, '%05dmask.%s'%(self.i, self.settings.imgending)))
        bpy.ops.clip.change_frame(frame=curr_frame+1)

    def setup(self, context):
        proj_dir = paths[-1]
        if proj_dir == '':
            raise ValueError('CleanPlateBlender path is empty.')
        self.frame_end = context.scene.frame_end
        self.settings = context.scene.cp_settings
        self.T = context.scene.frame_end-context.scene.frame_start
        assert self.T >= 12, 'At least 12 frames are required'
        self.W, self.H = self.hw  # context.scene.render.resolution_y, context.scene.render.resolution_x
        self.W, self.H = int(8*round(self.W/self.settings.downscale/8)), int(8*round(self.H/self.settings.downscale/8))
        self.complete = False
        # progress bar
        self.progress = 0
        self.wm = context.window_manager
        self.wm.progress_begin(0, 2+self.T*4)

        # create temporary directory
        self.tmp_dir = tempfile.TemporaryDirectory()

        # save Arguments
        self.args = Arguments()
        self.args.img_root = os.path.join(self.tmp_dir.name, 'frames')
        self.args.mask_root = os.path.join(self.tmp_dir.name, 'masks')
        self.args.flow_root = os.path.join(self.tmp_dir.name, 'Flow')
        self.args.inter_root = os.path.join(self.tmp_dir.name, 'interp')
        for d in (self.args.img_root, self.args.mask_root, self.args.flow_root):
            os.makedirs(d)
        self.args.imgending = self.settings.imgending
        self.args.dataset_root = self.tmp_dir.name
        self.args.frame_dir = self.args.img_root
        self.args.pretrained_model_liteflownet = os.path.join(proj_dir, 'weights', 'liteflownet.pth')
        self.args.pretrained_model_inpaint = os.path.join(proj_dir, 'weights', 'imagenet_deepfill.pth')
        self.args.PRETRAINED_MODEL_1 = os.path.join(proj_dir, 'weights', 'resnet101_movie.pth')
        self.args.n_threads = self.settings.n_threads
        self.args.output_root = self.tmp_dir.name
        self.args.output_root_propagation = self.settings.outpath
        self.args.img_size = [self.H, self.W]
        self.args.img_shape = self.args.img_size
        self.args.th_warp = self.settings.th_warp
        self.args.enlarge_mask = self.settings.mask_enlarge > 0
        self.args.enlarge_kernel = self.settings.mask_enlarge
        self.args.interpolation_steps = self.settings.steps
        # Load Model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args.device = device

        #mask = context.space_data.mask
        self.mask = bpy.data.masks[self.settings.mask_name]
        #co_tot, lhand_tot, rhand_tot = [], [], []
        bpy.ops.clip.change_frame(frame=context.scene.frame_start)
        self.cap = cv2.VideoCapture(self.movpath)
        self.cap.set(1, context.scene.frame_start-1)
        #T, H, W = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        self.i = -1
        self.state = 0

    def flow(self):
        if self.i == -1:
            # initialization
            self.args.data_list = infer_liteflownet.generate_flow_list(self.args.frame_dir)
            print('====> Loading', self.args.pretrained_model_liteflownet)
            self.Flownet = LiteFlowNet(self.args.pretrained_model_liteflownet)
            self.Flownet.to(self.args.device)
            self.Flownet.eval()

            dataset_ = FlowInfer.FlowInfer(self.args.data_list, size=self.args.img_size)
            self.flow_dataloader = iter(DataLoader(dataset_, batch_size=1, shuffle=False, num_workers=0))
        self.i += 1
        complete = False
        with torch.no_grad():
            try:
                f1, f2, output_path_ = next(self.flow_dataloader)
                f1 = f1.to(self.args.device)
                f2 = f2.to(self.args.device)

                flow = infer_liteflownet.estimate(self.Flownet, f1, f2)

                output_path = output_path_[0]
                output_file = os.path.dirname(output_path)
                os.makedirs(output_file, exist_ok=True)

                flow_numpy = flow[0].permute(1, 2, 0).data.cpu().numpy()
                cvb.write_flow(flow_numpy, output_path)
            except StopIteration:
                complete = True

        if self.i == len(self.flow_dataloader) - 1 or complete:
            print('LiteFlowNet Inference has been finished!')
            flow_list = [x for x in os.listdir(self.args.flow_root) if '.flo' in x]
            flow_start_no = min([int(x[:5]) for x in flow_list])
            del self.flow_dataloader, self.Flownet
            zero_flow = cvb.read_flow(os.path.join(self.args.flow_root, flow_list[0]))
            cvb.write_flow(zero_flow*0, os.path.join(self.args.flow_root, '%05d.rflo' % flow_start_no))
            self.args.DATA_ROOT = self.args.flow_root
            self.i = -1
            self.state += 1

    def flow_completion(self):
        if self.i == -1:
            data_list_dir = os.path.join(self.args.dataset_root, 'data')
            os.makedirs(data_list_dir, exist_ok=True)
            initial_data_list = os.path.join(data_list_dir, 'initial_test_list.txt')
            print('Generate datalist for initial step')
            data_list.gen_flow_initial_test_mask_list(flow_root=self.args.DATA_ROOT, output_txt_path=initial_data_list)
            self.args.EVAL_LIST = os.path.join(data_list_dir, 'initial_test_list.txt')

            self.args.output_root = os.path.join(self.args.dataset_root, 'Flow_res', 'initial_res')
            self.args.PRETRAINED_MODEL = self.args.PRETRAINED_MODEL_1

            if self.args.img_size is not None:
                self.args.IMAGE_SHAPE = [self.args.img_size[0] // 2, self.args.img_size[1] // 2]
                self.args.RES_SHAPE = self.args.IMAGE_SHAPE

            print('Flow Completion in First Step')
            self.args.MASK_ROOT = self.args.mask_root
            eval_dataset = FlowInitial.FlowSeq(self.args, isTest=True)
            self.flow_refinement_dataloader = iter(DataLoader(eval_dataset, batch_size=self.settings.batch_size, shuffle=False,
                                                              drop_last=False, num_workers=self.args.n_threads))
            if self.args.ResNet101:
                dfc_resnet101 = resnet_models.Flow_Branch(33, 2)
                self.dfc_resnet = nn.DataParallel(dfc_resnet101).to(self.args.device)
            else:
                dfc_resnet50 = resnet_models.Flow_Branch_Multi(input_chanels=33, NoLabels=2)
                self.dfc_resnet = nn.DataParallel(dfc_resnet50).to(self.args.device)
            self.dfc_resnet.eval()
            io.load_ckpt(self.args.PRETRAINED_MODEL, [('model', self.dfc_resnet)], strict=True)
            print('Load Pretrained Model from', self.args.PRETRAINED_MODEL)

        self.i += 1
        complete = False
        with torch.no_grad():
            try:
                item = next(self.flow_refinement_dataloader)
                input_x = item[0].to(self.args.device)
                flow_masked = item[1].to(self.args.device)
                mask = item[3].to(self.args.device)
                output_dir = item[4][0]

                res_flow = self.dfc_resnet(input_x)
                res_complete = res_flow * mask[:, 10:11, :, :] + flow_masked[:, 10:12, :, :] * (1. - mask[:, 10:11, :, :])

                output_dir_split = output_dir.split(',')
                output_file = os.path.join(self.args.output_root, output_dir_split[0])
                output_basedir = os.path.dirname(output_file)
                if not os.path.exists(output_basedir):
                    os.makedirs(output_basedir)
                res_save = res_complete[0].permute(1, 2, 0).contiguous().cpu().data.numpy()
                cvb.write_flow(res_save, output_file)
            except StopIteration:
                complete = True
        if self.i == len(self.flow_refinement_dataloader) - 1 or complete:
            self.args.flow_root = self.args.output_root
            del self.flow_refinement_dataloader, self.dfc_resnet
            self.i = -1
            self.state += 1

    def propagation(self):
        if self.i == -1:
            self.deepfill_model = frame_inpaint.DeepFillv1(pretrained_model=self.args.pretrained_model_inpaint, image_shape=self.args.img_shape)
            self.prop = propagation_inpaint.modal_propagation(self.args, frame_inpaint_model=self.deepfill_model)
            self.prop.file_ending = self.settings.imgending
        self.i += 1
        complete = self.prop.step()
        if self.prop.change_iter_num and not complete:
            self.prop.change_iter_num = False
            self.wm.progress_update(self.T*2)
        if complete:
            self.i = -1
            self.state += 1
            del self.deepfill_model

    def close(self):
        self.complete = True
        self.wm.progress_end()
        self.tmp_dir.cleanup()

    def execute(self, context):
        if self.state == -1:
            self.setup(context)
        elif self.state == 0:
            self.collect_next_frame()
        elif self.state == 1:
            self.flow()
        elif self.state == 2:
            self.flow_completion()
        elif self.state == 3:
            self.propagation()
        elif self.state == 4:
            self.state = -1
            self.close()
            return {'FINISHED'}
        self.progress += 1
        self.wm.progress_update(np.clip(self.progress, 0, 2+self.T*5))


class FrameInterpolator(CleanPlateMaker):

    def collect_next_frame(self):
        self.i += 1
        ret, frame = self.cap.read()
        curr_frame = bpy.context.scene.frame_current
        # state finished. return and go to next state
        if not ret or curr_frame == self.frame_end+1:
            self.cap.release()
            bpy.ops.clip.change_frame(frame=self.frame_end)
            self.i = -1
            self.state += 1
            return

        frame = cv2.resize(frame, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(self.args.img_root, '%05d.%s' % (curr_frame, self.settings.imgending)), frame)

        bpy.ops.clip.change_frame(frame=curr_frame+1)

    def interpolate(self):
        if self.i == -1:
            self.interpolator = interpolate.Interpolater(self.args)
        self.i += 1
        if self.interpolator.step():
            self.img_root = self.inter_root
            self.i = -1
            self.state += 1
            del self.interpolator

    def execute(self, context):
        if self.state == -1:
            self.setup(context)
        elif self.state == 0:
            self.collect_next_frame()
        elif self.state == 1:
            self.flow()
        elif self.state == 2:
            self.interpolate()
        elif self.state == 3:
            self.flow_completion()
        elif self.state == 4:
            self.propagation()
        elif self.state == 5:
            self.state = -1
            self.close()
            return {'FINISHED'}
        self.progress += 1
        self.wm.progress_update(np.clip(self.progress, 0, 2+self.T*5))


class OBJECT_OT_cleanplate(Operator):
    bl_idname = "object.cleanplate"
    bl_label = ""
    bl_description = "Removes everything in the selected Mask"

    _updating = False
    _calcs_done = True
    _timer = None

    def modal(self, context, event):
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self._calcs_done = True
        elif event.type == 'TIMER' and not self._updating and not self._calcs_done:
            self._updating = True
            #frame_end = context.scene.frame_end
            # if bpy.context.scene.frame_current < frame_end:
            ret = self.cpm.execute(context)
            if type(ret) == set:
                self._calcs_done = True

            self._updating = False
        if self._calcs_done:
            self.cancel(context)
        return {'PASS_THROUGH'}

    def execute(self, context):
        clip = context.space_data.clip
        self.cpm = CleanPlateMaker()
        self.cpm.movpath = bpy.path.abspath(clip.filepath)
        self.cpm.hw = clip.size
        self.cpm.set_coordinate_transform()

        self._calcs_done = False
        context.window_manager.modal_handler_add(self)
        self._updating = False
        self._timer = context.window_manager.event_timer_add(.01, window=context.window)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
            if self.cpm.complete:
                self.report({'INFO'}, "Inpainting complete.")
            del self.cpm
        return {'CANCELLED'}


class PANEL0_PT_settings(Panel):
    bl_label = "Settings"
    bl_idname = "PANEL0_PT_settings"
    bl_space_type = 'CLIP_EDITOR'
    bl_region_type = 'UI'
    bl_category = "CleanPlate"

    @classmethod
    def poll(cls, context):
        return (context.area.spaces.active.clip is not None)

    # Draw UI
    def draw(self, context):
        settings = context.scene.cp_settings
        layout = self.layout
        layout.use_property_split = True  # Active single-column layout
        layout.prop(settings, 'downscale')
        layout.prop(settings, 'imgending', icon='FILE_IMAGE')
        layout.prop(settings, 'outpath')
        #layout.prop(settings, 'n_threads')
        #layout.prop(settings, 'batch_size')
        layout.prop(settings, 'th_warp')


class PANEL0_PT_cleanplate(Panel):
    bl_label = "Inpainting"
    bl_idname = "PANEL0_PT_cleanplate"
    bl_space_type = 'CLIP_EDITOR'
    bl_region_type = 'UI'
    bl_category = "CleanPlate"

    @classmethod
    def poll(cls, context):
        return (context.area.spaces.active.clip is not None)

    # Draw UI
    def draw(self, context):
        settings = context.scene.cp_settings
        layout = self.layout
        layout.use_property_split = True  # Active single-column layout
        layout.prop(settings, 'mask_name')
        layout.prop(settings, 'downscale')
        layout.prop(settings, 'mask_enlarge')
        layout.separator()
        row = layout.row()
        row.operator("object.cleanplate", text="Create Clean Plate")



class PANEL0_PT_interpolation(Panel):
    bl_label = "Frame Interpolation"
    bl_idname = "PANEL0_PT_interpolation"
    bl_space_type = 'CLIP_EDITOR'
    bl_region_type = 'UI'
    bl_category = "CleanPlate"

    @classmethod
    def poll(cls, context):
        return (context.area.spaces.active.clip is not None)

    # Draw UI
    def draw(self, context):
        settings = context.scene.cp_settings
        layout = self.layout
        layout.use_property_split = True  # Active single-column layout
        layout.prop(settings, 'steps')
        layout.separator()
        row = layout.row()
        row.operator("object.interpolate", text="Generate New Frames")


class OBJECT_OT_interpolation(Operator):
    bl_idname = "object.interpolate"
    bl_label = ""
    bl_description = "Generates new frames in between existing frames"

    _updating = False
    _calcs_done = True
    _timer = None

    def modal(self, context, event):
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self._calcs_done = True
        elif event.type == 'TIMER' and not self._updating and not self._calcs_done:
            self._updating = True
            #frame_end = context.scene.frame_end
            # if bpy.context.scene.frame_current < frame_end:
            ret = self.cpm.execute(context)
            if type(ret) == set:
                self._calcs_done = True

            self._updating = False
        if self._calcs_done:
            self.cancel(context)
        return {'PASS_THROUGH'}

    def execute(self, context):
        clip = context.space_data.clip
        self.cpm = FrameInterpolator()
        self.cpm.movpath = bpy.path.abspath(clip.filepath)
        self.cpm.hw = clip.size
        self.cpm.set_coordinate_transform()

        self._calcs_done = False
        context.window_manager.modal_handler_add(self)
        self._updating = False
        self._timer = context.window_manager.event_timer_add(.01, window=context.window)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
            if self.cpm.complete:
                self.report({'INFO'}, "Inpainting complete.")
            del self.cpm
        return {'CANCELLED'}


classes = (OBJECT_OT_cleanplate,PANEL0_PT_settings, PANEL0_PT_cleanplate, Settings, PANEL0_PT_interpolation, OBJECT_OT_interpolation)


def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)
    bpy.types.Scene.cp_settings = PointerProperty(type=Settings)


def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)
    del bpy.types.Scene.cp_settings


if __name__ == "__main__":
    register()
