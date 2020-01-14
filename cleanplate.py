
from __future__ import print_function
from mask_spline import *
import bpy
from bpy.types import Operator, Panel, PropertyGroup, WindowManager
from bpy.props import PointerProperty, StringProperty, IntProperty, FloatProperty, BoolProperty
import sys
import cv2
import torch.nn as nn
import torch
import os
paths = [
    r'C:\Users\user\AppData\Local\Continuum\anaconda3\lib\site-packages',
    r'C:\Users\user\Desktop\ML\CleanPlateBlender'
]
for p in paths:
    sys.path.insert(0, p)
from models.OPN import OPN
from models.TCN import TCN

bl_info = {
    'blender': (2, 80, 0),
    'name': 'CleanPlate',
    'category': 'Motion Tracking',
    'location': 'Masking> Movie Clip Editor > CleanPlate',
    'author': 'Lukas Blecher'
}


class Settings(PropertyGroup):

    maxnum: IntProperty(
        name="Directions",
        description="The lower this value is the more points will be created",
        default=3,
        min=1,
        max=5
    )

    maxlen: IntProperty(
        name="Max. Length",
        description="The maximum amount of pixels a mask line segment is tracing",
        default=150,
        min=1
    )

    threshold: IntProperty(
        name="Treshold",
        description="The amount of points that can point in a different direction\nbefore a new segment is created",
        default=10,
        min=0
    )

    my_float: FloatProperty(
        name="Float Value",
        description="A float property",
        default=23.7,
        min=0.01,
        max=30.0
    )

    change_layer: BoolProperty(
        name="Change Layer",
        description="Change the active Mask Layer according to the frame\nwhen moving the along the timeline",
        default=True
    )


class CleanPlateMaker:
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

    def cleanplate(self, context, model, state, movpath):
        mask = context.space_data.mask
        settings = context.scene.settings
        layer = mask.layers.active
        maskSplines = layer.splines
        #co_tot, lhand_tot, rhand_tot = [], [], []
        bpy.ops.clip.change_frame(frame=1)
        cap = cv2.VideoCapture(movpath)
        #cap.set(1, framenum-1)
        #T, H, W = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        T, H, W = context.scene.frame_end-context.scene.frame_start, context.scene.render.resolution_y, context.scene.render.resolution_x

        frames = np.empty((T, H, W, 3), dtype=np.float32)
        holes = np.empty((T, H, W, 1), dtype=np.float32)
        dists = np.empty((T, H, W, 1), dtype=np.float32)
        i = -1
        while cap.isOpened():
            i += 1
            ret, frame = cap.read()
            if not ret or i == T:
                cap.release()
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, dsize=(W, H), interpolation=cv2.INTER_LINEAR)
            frames[i] = np.array(frame)/255
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
                raw_mask = crl2mask(crl, int(self.hw[0]), int(self.hw[1])).astype(np.uint8)
                raw_mask = cv2.resize(raw_mask, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
                raw_mask = cv2.dilate(raw_mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
                holes[i, :, :, 0] = raw_mask.astype(np.float32)
                # dist
                dists[i, :, :, 0] = cv2.distanceTransform(raw_mask, cv2.DIST_L2, maskSize=5)
            bpy.ops.clip.change_frame(frame=bpy.context.scene.frame_current+1)
        

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
            frame_end = context.scene.frame_end
            if bpy.context.scene.frame_current < frame_end-1:
                ret = self.cpm.cleanplate(context, self.model, self.state, self.cpm.movpath)
                if type(ret) == set:
                    self._calcs_done = True
                else:
                    self.model = ret[0]
                    self.state = ret[1]
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
        proj_dir = paths[-1]
        if proj_dir == '':
            raise ValueError('CleanPlateBlender path is empty.')
        
        # Load Model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cpm.model = nn.DataParallel(OPN()).to(device)
        self.cpm.model.load_state_dict(torch.load(os.path.join(proj_dir, 'OPN.pth')), strict=False)
        self.cpm.model.eval()

        self.cpm.pp_model = nn.DataParallel(TCN()).to(device)
        self.cpm.pp_model.load_state_dict(torch.load(os.path.join(proj_dir, 'TCN.pth')), strict=False)
        self.cpm.pp_model.eval()

        self._calcs_done = False
        context.window_manager.modal_handler_add(self)
        self._updating = False
        self._timer = context.window_manager.event_timer_add(.01, window=context.window)
        return {'FINISHED'}

    def cancel(self, context):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
            del self.model
            del self.state
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
        settings = context.scene.settings
        layout = self.layout
        layout.use_property_split = True  # Active single-column layout
        # track masks operators
        c = layout.column()
        row = c.row()
        split = row.split(factor=0.3)
        c = split.column()
        c.label(text="Track:")
        split = split.split()
        c = split.row()
        c.operator("object.cleanplate", icon="TRACKING_FORWARDS")
        row = layout.column()
        layout.prop(settings, 'maxlen')
        layout.prop(settings, 'threshold')
        layout.prop(settings, 'maxnum')
        layout.prop(settings, 'change_layer')

        layout.separator()


classes = (OBJECT_OT_cleanplate, OBJECT_OT_clear_forwards, OBJECT_OT_clear_backwards, PANEL0_PT_cleanplate, Settings)


def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)
    bpy.types.Scene.settings = PointerProperty(type=Settings)


def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)
    del bpy.types.Scene.settings


if __name__ == "__main__":
    register()
