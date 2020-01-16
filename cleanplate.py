
import bpy
from bpy.types import Operator, Panel, PropertyGroup, WindowManager
from bpy.props import PointerProperty, StringProperty, IntProperty, FloatProperty, BoolProperty, EnumProperty
import sys
import os
paths = [
    r'C:\Users\user\AppData\Local\Continuum\anaconda3\lib\site-packages',
    r'C:\Users\user\Desktop\ML\CleanPlateBlender',
    #os.path.dirname(os.path.realpath(__file__))
]
for p in paths:
    sys.path.insert(0, p)

import cv2
import torch.nn as nn
import torch

from models.OPN import OPN
from models.TCN import TCN
from mask_spline import *

bl_info = {
    'blender': (2, 80, 0),
    'name': 'CleanPlate',
    'category': 'Motion Tracking',
    'location': 'Masking > Movie Clip Editor > CleanPlate',
    'author': 'Lukas Blecher'
}


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
        items=mask_name_callback
    )

    memevery: IntProperty(
        name="Mem. every",
        description="Memorize every nth frame",
        default=5,
        min=1
    )

    outpath: StringProperty(
        name="Output Directory",
        description="Where to save the inpainted images",
        default='/tmp',
        maxlen=1024,
        subtype='DIR_PATH'
    )

    imgending: StringProperty(
        name="File Format",
        description="File Format for the inpainted images",
        default='png'
    )

    downscale: FloatProperty(
        name="Downscaling Factor",
        description="How much to downscale the image",
        default=1,
        min=1
    )

    change_layer: BoolProperty(
        name="Change Layer",
        description="Change the active Mask Layer according to the frame\nwhen moving the along the timeline",
        default=True
    )


class CleanPlateMaker:
    state = -1

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
        # state finished. return and go to next state
        if not ret or self.i == self.T:
            self.cap.release()
            self.i = -1
            self.state += 1
            self.frames = torch.from_numpy(np.transpose(self.frames, (3, 0, 1, 2)).copy()).float()
            self.holes = torch.from_numpy(np.transpose(self.holes, (3, 0, 1, 2)).copy()).float()
            self.dists = torch.from_numpy(np.transpose(self.dists, (3, 0, 1, 2)).copy()).float()
            # remove hole
            self.frames = self.frames * (1-self.holes) + self.holes*torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
            # valids area
            self.valids = 1-self.holes
            # unsqueeze to batch 1
            self.frames = self.frames.unsqueeze(0)
            self.holes = self.holes.unsqueeze(0)
            self.dists = self.dists.unsqueeze(0)
            self.valids = self.valids.unsqueeze(0)
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        self.frames[self.i] = np.array(frame)/255
        raw_mask = np.zeros((self.H, self.W), dtype=np.uint8)
        maskSplines = self.mask.layers.active.splines
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
            raw_mask += crl2mask(crl, self.hw[1], self.hw[0], downscale=self.settings.downscale).astype(np.uint8)
        raw_mask = np.clip(raw_mask, 0, 1)
        #canvas = Image.fromarray(raw_mask*255)
        raw_mask = cv2.resize(raw_mask, dsize=(self.W, self.H), interpolation=cv2.INTER_NEAREST)
        raw_mask = cv2.dilate(raw_mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
        #canvas.save(os.path.join(self.settings.outpath, '%05dmask.%s'%(self.i, self.settings.imgending)))
        self.holes[self.i, :, :, 0] = raw_mask.astype(np.float32)
        # dist
        self.dists[self.i, :, :, 0] = cv2.distanceTransform(raw_mask, cv2.DIST_L2, maskSize=5)
        bpy.ops.clip.change_frame(frame=bpy.context.scene.frame_current+1)

    def setup(self, context):
        proj_dir = paths[-1]
        if proj_dir == '':
            raise ValueError('CleanPlateBlender path is empty.')

        self.settings = context.scene.cp_settings
        self.T = context.scene.frame_end-context.scene.frame_start
        self.W, self.H = self.hw  # context.scene.render.resolution_y, context.scene.render.resolution_x
        self.W, self.H = int(self.W//self.settings.downscale), int(self.H//self.settings.downscale)
        self.frames = np.empty((self.T, self.H, self.W, 3), dtype=np.float32)
        self.holes = np.empty((self.T, self.H, self.W, 1), dtype=np.float32)
        self.dists = np.empty((self.T, self.H, self.W, 1), dtype=np.float32)

        # progress bar
        self.progress = 0
        self.wm = context.window_manager
        self.wm.progress_begin(0, 2+self.T*4)

        # Load Model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = nn.DataParallel(OPN()).to(device)
        self.model.load_state_dict(torch.load(os.path.join(proj_dir, 'weights', 'OPN.pth')), strict=False)
        self.model.eval()

        self.pp_model = nn.DataParallel(TCN()).to(device)
        self.pp_model.load_state_dict(torch.load(os.path.join(proj_dir, 'weights', 'TCN.pth')), strict=False)
        self.pp_model.eval()
        #mask = context.space_data.mask
        self.mask = bpy.data.masks[self.settings.mask_name]
        #co_tot, lhand_tot, rhand_tot = [], [], []
        bpy.ops.clip.change_frame(frame=context.scene.frame_start)
        self.cap = cv2.VideoCapture(self.movpath)
        self.cap.set(1, context.scene.frame_start-1)
        #T, H, W = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        self.i = -1
        self.state = 0

    def memory_encoding(self):
        self.comps = torch.zeros_like(self.frames)
        self.ppeds = torch.zeros_like(self.frames)
        # memory encoding
        self.midx = list(range(0, self.T, self.settings.memevery))
        with torch.no_grad():
            self.mkey, self.mval, self.mhol = self.model(self.frames[:, :, self.midx], self.valids[:, :, self.midx], self.dists[:, :, self.midx])
        self.state += 1

    def inpainting(self):
        self.i += 1
        f = self.i
        # memory selection
        if f in self.midx:
            ridx = [i for i in range(len(self.midx)) if i != int(f/self.settings.memevery)]
        else:
            ridx = list(range(len(self.midx)))

        fkey, fval, fhol = self.mkey[:, :, ridx], self.mval[:, :, ridx], self.mhol[:, :, ridx]
        # inpainting..
        for r in range(999):
            if r == 0:
                self.comp = self.frames[:, :, f]
                self.dist = self.dists[:, :, f]
            with torch.no_grad():
                self.comp, self.dist = self.model(fkey, fval, fhol, self.comp, self.valids[:, :, f], self.dist)
            # update
            self.comp, self.dist = self.comp.detach(), self.dist.detach()
            if torch.sum(self.dist).item() == 0:
                break
        self.comps[:, :, f] = self.comp
        if f == self.T - 1:
            self.i = -1
            self.state += 1
            self.ppeds[:, :, 0] = self.comps[:, :, 0]
            self.hidden = None

    def postprocess(self):
        self.i += 1
        f = self.i
        with torch.no_grad():
            self.pped,  self.hidden = self.pp_model(self.ppeds[:, :, f-1], self.holes[:, :, f-1], self.comps[:, :, f], self.holes[:, :, f], self.hidden)
            self.ppeds[:, :, f] = self.pped
        if f == self.T - 1:
            self.i = -1
            self.state += 1

    def save(self, context):
        self.i += 1
        f = self.i
        canvas = (self.ppeds[0, :, f].permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)
        save_path = self.settings.outpath
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        canvas = Image.fromarray(canvas)
        canvas.save(os.path.join(save_path, '%05d.%s' % (f+context.scene.frame_start, self.settings.imgending)))
        if f == self.T - 1:
            self.i = -1
            self.state += 1

    def cleanplate(self, context):
        if self.state == -1:
            self.setup(context)
        elif self.state == 0:
            self.collect_next_frame()
        elif self.state == 1:
            self.memory_encoding()
        elif self.state == 2:
            self.inpainting()
        elif self.state == 3:
            self.postprocess()
        elif self.state == 4:
            self.save(context)
        elif self.state == 5:
            self.state = -1
            self.wm.progress_end()
            return {'FINISHED'}
        self.progress += 1
        self.wm.progress_update(np.clip(self.progress, 0, 2+self.T*4))


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
            ret = self.cpm.cleanplate(context)
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
            del self.cpm
        return {'CANCELLED'}


class PANEL0_PT_cleanplate(Panel):
    bl_label = "Mask Tracking"
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
        layout.prop(settings, 'memevery')
        layout.prop(settings, 'imgending', icon='FILE_IMAGE')
        layout.prop(settings, 'outpath')
        layout.separator()
        row = layout.row()
        row.operator("object.cleanplate", text="Create Clean Plate")


classes = (OBJECT_OT_cleanplate, PANEL0_PT_cleanplate, Settings)


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
