"""
Based on https://github.com/seoungwugoh/ivs-demo

The entry point for the user interface
It is terribly long... GUI code is hard to write!
"""

import sys
import os
from os import path
import functools
from argparse import ArgumentParser

import cv2
from PIL import Image
import numpy as np
import torch
from collections import deque

from PyQt5.QtWidgets import (QWidget, QApplication, QComboBox, 
    QHBoxLayout, QLabel, QPushButton, QTextEdit, 
    QPlainTextEdit, QVBoxLayout, QSizePolicy, QButtonGroup, QSlider, 
    QShortcut, QRadioButton, QProgressBar, QFileDialog)

from PyQt5.QtGui import QPixmap, QKeySequence, QImage, QTextCursor
from PyQt5.QtCore import Qt, QTimer 
from PyQt5 import QtCore

from inference_core import InferenceCore
from interact.s2m_controller import S2MController
from interact.fbrs_controller import FBRSController
from model.propagation.prop_net import PropagationNetwork
from model.fusion_net import FusionNet
from model.s2m.s2m_network import deeplabv3plus_resnet50 as S2M
from util.tensor_util import unpad_3dim
from util.palette import pal_color_map

from interact.interactive_utils import *
from interact.interaction import *
from interact.timer import Timer

torch.set_grad_enabled(False)

# DAVIS palette
palette = pal_color_map()

class App(QWidget):
    def __init__(self, prop_net, fuse_net, s2m_ctrl:S2MController, fbrs_ctrl:FBRSController, 
                    images, masks, num_objects, mem_freq, mem_profile, sv_path=None):
        super().__init__()

        self.images = images
        self.masks = masks
        self.num_objects = num_objects
        self.s2m_controller = s2m_ctrl
        self.fbrs_controller = fbrs_ctrl
        self.processor = InferenceCore(prop_net, fuse_net, images_to_torch(images, device='cpu'),
                         num_objects, mem_freq=mem_freq, mem_profile=mem_profile)

        self.num_frames, self.height, self.width = self.images.shape[:3]

        # IOU computation
        if self.masks is not None:
            self.ious = np.zeros(self.num_frames)
            self.iou_curve = []

        # set window
        self.setWindowTitle('MiVOS')
        self.setGeometry(100, 100, self.width, self.height+100)

        # some buttons
        self.play_button = QPushButton('Play')
        self.play_button.clicked.connect(self.on_play)
        self.run_button = QPushButton('Propagate')
        self.run_button.clicked.connect(self.on_run)
        self.commit_button = QPushButton('Commit')
        self.commit_button.clicked.connect(self.on_commit)

        self.undo_button = QPushButton('Undo')
        self.undo_button.clicked.connect(self.on_undo)
        self.reset_button = QPushButton('Reset Frame')
        self.reset_button.clicked.connect(self.on_reset)
        self.save_button = QPushButton('Save')
        self.save_button.clicked.connect(self.save)

        # LCD
        self.lcd = QTextEdit()
        self.lcd.setReadOnly(True)
        self.lcd.setMaximumHeight(28)
        self.lcd.setMaximumWidth(120)
        self.lcd.setText('{: 4d} / {: 4d}'.format(0, self.num_frames-1))

        # timeline slider
        self.tl_slider = QSlider(Qt.Horizontal)
        self.tl_slider.valueChanged.connect(self.tl_slide)
        self.tl_slider.setMinimum(0)
        self.tl_slider.setMaximum(self.num_frames-1)
        self.tl_slider.setValue(0)
        self.tl_slider.setTickPosition(QSlider.TicksBelow)
        self.tl_slider.setTickInterval(1)
        
        # brush size slider
        self.brush_label = QLabel()
        self.brush_label.setAlignment(Qt.AlignCenter)
        self.brush_label.setMinimumWidth(100)
        
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.valueChanged.connect(self.brush_slide)
        self.brush_slider.setMinimum(1)
        self.brush_slider.setMaximum(100)
        self.brush_slider.setValue(3)
        self.brush_slider.setTickPosition(QSlider.TicksBelow)
        self.brush_slider.setTickInterval(2)
        self.brush_slider.setMinimumWidth(300)

        # combobox
        self.combo = QComboBox(self)
        self.combo.addItem("davis")
        self.combo.addItem("fade")
        self.combo.addItem("light")
        self.combo.currentTextChanged.connect(self.set_viz_mode)

        # Radio buttons for type of interactions
        self.curr_interaction = 'Click'
        self.interaction_group = QButtonGroup()
        self.radio_fbrs = QRadioButton('Click')
        self.radio_s2m = QRadioButton('Scribble')
        self.radio_free = QRadioButton('Free')
        self.interaction_group.addButton(self.radio_fbrs)
        self.interaction_group.addButton(self.radio_s2m)
        self.interaction_group.addButton(self.radio_free)
        self.radio_fbrs.toggled.connect(self.interaction_radio_clicked)
        self.radio_s2m.toggled.connect(self.interaction_radio_clicked)
        self.radio_free.toggled.connect(self.interaction_radio_clicked)
        self.radio_fbrs.toggle()

        # Main canvas -> QLabel
        self.main_canvas = QLabel()
        self.main_canvas.setSizePolicy(QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        self.main_canvas.setAlignment(Qt.AlignCenter)
        self.main_canvas.setMinimumSize(100, 100)

        self.main_canvas.mousePressEvent = self.on_press
        self.main_canvas.mouseMoveEvent = self.on_motion
        self.main_canvas.setMouseTracking(True) # Required for all-time tracking
        self.main_canvas.mouseReleaseEvent = self.on_release

        # Minimap -> Also a QLbal
        self.minimap = QLabel()
        self.minimap.setSizePolicy(QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        self.minimap.setAlignment(Qt.AlignTop)
        self.minimap.setMinimumSize(100, 100)

        # Zoom-in buttons
        self.zoom_p_button = QPushButton('Zoom +')
        self.zoom_p_button.clicked.connect(self.on_zoom_plus)
        self.zoom_m_button = QPushButton('Zoom -')
        self.zoom_m_button.clicked.connect(self.on_zoom_minus)
        self.finish_local_button = QPushButton('Finish Local')
        self.finish_local_button.clicked.connect(self.on_finish_local)
        self.finish_local_button.setDisabled(True)

        # Console on the GUI
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(100)
        self.console.setMaximumHeight(100)

        # progress bar
        self.progress = QProgressBar(self)
        self.progress.setGeometry(0, 0, 300, 25)
        self.progress.setMinimumWidth(300)
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setFormat('Idle')
        self.progress.setStyleSheet("QProgressBar{color: black;}")
        self.progress.setAlignment(Qt.AlignCenter)

        # navigator
        navi = QHBoxLayout()
        navi.addWidget(self.lcd)
        navi.addWidget(self.play_button)

        interact_subbox = QVBoxLayout()
        interact_topbox = QHBoxLayout()
        interact_botbox = QHBoxLayout()
        interact_topbox.setAlignment(Qt.AlignCenter)
        interact_topbox.addWidget(self.radio_s2m)
        interact_topbox.addWidget(self.radio_fbrs)
        interact_topbox.addWidget(self.radio_free)
        interact_topbox.addWidget(self.brush_label)
        interact_botbox.addWidget(self.brush_slider)
        interact_subbox.addLayout(interact_topbox)
        interact_subbox.addLayout(interact_botbox)
        navi.addLayout(interact_subbox)

        navi.addStretch(1)
        navi.addWidget(self.undo_button)
        navi.addWidget(self.reset_button)

        navi.addStretch(1)
        navi.addWidget(self.progress)
        navi.addWidget(QLabel('Overlay Mode'))
        navi.addWidget(self.combo)
        navi.addStretch(1)
        navi.addWidget(self.commit_button)
        navi.addWidget(self.run_button)
        navi.addWidget(self.save_button)

        # Drawing area, main canvas and minimap
        draw_area = QHBoxLayout()
        draw_area.addWidget(self.main_canvas, 4)

        # Minimap area
        minimap_area = QVBoxLayout()
        minimap_area.setAlignment(Qt.AlignTop)
        mini_label = QLabel('Minimap')
        mini_label.setAlignment(Qt.AlignTop)
        minimap_area.addWidget(mini_label)
        # Minimap zooming
        minimap_ctrl = QHBoxLayout()
        minimap_ctrl.setAlignment(Qt.AlignTop)
        minimap_ctrl.addWidget(self.zoom_p_button)
        minimap_ctrl.addWidget(self.zoom_m_button)
        minimap_ctrl.addWidget(self.finish_local_button)
        minimap_area.addLayout(minimap_ctrl)
        minimap_area.addWidget(self.minimap)
        minimap_area.addWidget(QLabel('Overall procedure: '))
        minimap_area.addWidget(QLabel('1. Label a frame (all objects) with whatever means'))
        minimap_area.addWidget(QLabel('2. Propagate'))
        minimap_area.addWidget(QLabel('3. Find a frame with error, correct it and proagatte again'))
        minimap_area.addWidget(QLabel('4. Repeat'))
        minimap_area.addWidget(QLabel('Tips: '))
        minimap_area.addWidget(QLabel('1: Use Ctrl+Left-click to drag-select a local control region.'))
        minimap_area.addWidget(QLabel('Click finish local to go back.'))
        minimap_area.addWidget(QLabel('2: Use Right-click to label background.'))
        minimap_area.addWidget(QLabel('3: Use Num-keys to change the object id. '))
        minimap_area.addWidget(QLabel('(1-Red, 2-Green, 3-Blue, ...)'))
        minimap_area.addWidget(QLabel('4: \"Commit\" only works for S2M, it clears the buffer.'))
        minimap_area.addWidget(self.console)

        draw_area.addLayout(minimap_area, 1)

        layout = QVBoxLayout()
        layout.addLayout(draw_area)
        layout.addWidget(self.tl_slider)
        layout.addLayout(navi)
        self.setLayout(layout)

        # timer
        self.timer = QTimer()
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(self.on_time)

        # Local mode related states
        self.ctrl_key = False
        self.in_local_mode = False
        self.local_bb = None
        self.local_interactions = {}
        self.this_local_interactions = []
        self.local_interaction = None

        # initialize visualization
        self.viz_mode = 'davis'
        self.current_mask = np.zeros((self.num_frames, self.height, self.width), dtype=np.uint8)
        self.vis_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.vis_alpha = np.zeros((self.height, self.width, 1), dtype=np.float32)
        self.brush_vis_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.brush_vis_alpha = np.zeros((self.height, self.width, 1), dtype=np.float32)
        self.vis_hist = deque(maxlen=100)
        self.cursur = 0
        self.on_showing = None

        # initialize local visualization (which is mostly unknown at this point)
        self.local_vis_map = None
        self.local_vis_alpha = None
        self.local_brush_vis_map = None
        self.local_brush_vis_alpha = None
        self.local_vis_hist = deque(maxlen=100)

        # Zoom parameters
        self.zoom_pixels = 150
        
        # initialize action
        self.interactions = {}
        self.interactions['interact'] = [[] for _ in range(self.num_frames)]
        self.interactions['annotated_frame'] = []
        self.this_frame_interactions = []
        self.interaction = None
        self.reset_this_interaction()
        self.pressed = False
        self.right_click = False
        self.ctrl_size = False
        self.current_object = 1
        self.last_ex = self.last_ey = 0

        # Objects shortcuts
        for i in range(1, num_objects+1):
            QShortcut(QKeySequence(str(i)), self).activated.connect(functools.partial(self.hit_number_key, i))

        # <- and -> shortcuts
        QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(self.on_prev)
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(self.on_next)
        
        # Mask saving
        # QShortcut(QKeySequence('s'), self).activated.connect(self.save)
        # QShortcut(QKeySequence('l'), self).activated.connect(self.debug_pressed)
        self.sv_path = sv_path

        self.interacted_mask = None

        self.show_current_frame()
        self.show()

        self.waiting_to_start = True
        self.global_timer = Timer().start()
        self.algo_timer = Timer()
        self.user_timer = Timer()
        self.console_push_text('Initialized.')
 
    def resizeEvent(self, event):
        self.show_current_frame()

    def save(self):
        folder_path = str(QFileDialog.getExistingDirectory(self, "Select Save Directory")) if self.sv_path is None else self.sv_path

        self.console_push_text('Saving masks and overlays...')
        mask_dir = path.join(folder_path, 'mask')
        overlay_dir = path.join(folder_path, 'overlay')

        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)

        for i in range(self.num_frames):
            # Save mask
            mask = Image.fromarray(self.current_mask[i]).convert('P')
            mask.putpalette(palette)
            mask.save(os.path.join(mask_dir, '{:05d}.png'.format(i)))

            # Save overlay
            overlay = overlay_davis(self.images[i], self.current_mask[i]) 
            overlay = Image.fromarray(overlay)
            overlay.save(os.path.join(overlay_dir, '{:05d}.png'.format(i)))
        self.console_push_text('Done.')
        self.sv_path = folder_path

    def console_push_text(self, text):
        text = '[A: %s, U: %s]: %s' % (self.algo_timer.format(), self.user_timer.format(), text)
        self.console.appendPlainText(text)
        self.console.moveCursor(QTextCursor.End)
        print(text)

    def interaction_radio_clicked(self, event):
        self.last_interaction = self.curr_interaction
        if self.radio_s2m.isChecked():
            self.curr_interaction = 'Scribble'
            self.brush_size = 3
            self.brush_slider.setDisabled(True)
        elif self.radio_fbrs.isChecked():
            self.curr_interaction = 'Click'
            self.brush_size = 3
            self.brush_slider.setDisabled(True)
        elif self.radio_free.isChecked():
            self.brush_slider.setDisabled(False)
            self.brush_slide()
            self.curr_interaction = 'Free'
        if self.curr_interaction == 'Scribble':
            self.commit_button.setEnabled(True)
        else:
            self.commit_button.setEnabled(False)

        # if self.last_interaction != self.curr_interaction:
            # self.console_push_text('Interaction changed to ' + self.curr_interaction + '.')

    def compose_current_im(self):
        if self.in_local_mode:
            if self.viz_mode == 'fade':
                self.viz = overlay_davis_fade(self.local_np_im, self.local_np_mask) 
            elif self.viz_mode == 'davis':
                self.viz = overlay_davis(self.local_np_im, self.local_np_mask) 
            elif self.viz_mode == 'light':
                self.viz = overlay_davis(self.local_np_im, self.local_np_mask, 0.9) 
            else:
                raise NotImplementedError
        else:
            if self.viz_mode == 'fade':
                self.viz = overlay_davis_fade(self.images[self.cursur], self.current_mask[self.cursur]) 
            elif self.viz_mode == 'davis':
                self.viz = overlay_davis(self.images[self.cursur], self.current_mask[self.cursur]) 
            elif self.viz_mode == 'light':
                self.viz = overlay_davis(self.images[self.cursur], self.current_mask[self.cursur], 0.9)
            else:
                raise NotImplementedError

    def update_interact_vis(self):
        # Update the interactions without re-computing the overlay
        height, width, channel = self.viz.shape
        bytesPerLine = 3 * width

        if self.in_local_mode:
            vis_map = self.local_vis_map
            vis_alpha = self.local_vis_alpha
            brush_vis_map = self.local_brush_vis_map
            brush_vis_alpha = self.local_brush_vis_alpha
        else:
            vis_map = self.vis_map
            vis_alpha = self.vis_alpha
            brush_vis_map = self.brush_vis_map
            brush_vis_alpha = self.brush_vis_alpha

        self.viz_with_stroke = self.viz*(1-vis_alpha) + vis_map*vis_alpha
        self.viz_with_stroke = self.viz_with_stroke*(1-brush_vis_alpha) + brush_vis_map*brush_vis_alpha
        self.viz_with_stroke = self.viz_with_stroke.astype(np.uint8)

        qImg = QImage(self.viz_with_stroke.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.main_canvas.setPixmap(QPixmap(qImg.scaled(self.main_canvas.size(),
                Qt.KeepAspectRatio, Qt.FastTransformation)))

        self.main_canvas_size = self.main_canvas.size()
        self.image_size = qImg.size()

    def update_minimap(self):
        # Limit it within the valid range
        if self.in_local_mode:
            if self.minimap_in_local_drawn:
                # Do not redraw
                return
            self.minimap_in_local_drawn = True
            patch = self.minimap_in_local.astype(np.uint8)
        else:
            ex, ey = self.last_ex, self.last_ey
            r = self.zoom_pixels//2
            ex = int(round(max(r, min(self.width-r, ex))))
            ey = int(round(max(r, min(self.height-r, ey))))

            patch = self.viz_with_stroke[ey-r:ey+r, ex-r:ex+r, :].astype(np.uint8)

        height, width, channel = patch.shape
        bytesPerLine = 3 * width
        qImg = QImage(patch.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.minimap.setPixmap(QPixmap(qImg.scaled(self.minimap.size(),
                Qt.KeepAspectRatio, Qt.FastTransformation)))

    def show_current_frame(self):
        # Re-compute overlay and show the image
        self.compose_current_im()
        self.update_interact_vis()
        self.update_minimap()
        self.lcd.setText('{: 3d} / {: 3d}'.format(self.cursur, self.num_frames-1))
        self.tl_slider.setValue(self.cursur)

    def get_scaled_pos(self, x, y):
        # Un-scale and un-pad the label coordinates into image coordinates
        oh, ow = self.image_size.height(), self.image_size.width()
        nh, nw = self.main_canvas_size.height(), self.main_canvas_size.width()

        h_ratio = nh/oh
        w_ratio = nw/ow
        dominate_ratio = min(h_ratio, w_ratio)

        # Solve scale
        x /= dominate_ratio
        y /= dominate_ratio

        # Solve padding
        fh, fw = nh/dominate_ratio, nw/dominate_ratio
        x -= (fw-ow)/2
        y -= (fh-oh)/2

        if self.in_local_mode:
            x = max(0, min(self.local_width-1, x))
            y = max(0, min(self.local_height-1, y))
        else:
            x = max(0, min(self.width-1, x))
            y = max(0, min(self.height-1, y))

        # return int(round(x)), int(round(y))
        return x, y

    def clear_visualization(self):
        if self.in_local_mode:
            self.local_vis_map.fill(0)
            self.local_vis_alpha.fill(0)
            self.local_vis_hist.clear()
            self.local_vis_hist.append((self.local_vis_map.copy(), self.local_vis_alpha.copy()))
        else:
            self.vis_map.fill(0)
            self.vis_alpha.fill(0)
            self.vis_hist.clear()
            self.vis_hist.append((self.vis_map.copy(), self.vis_alpha.copy()))

    def reset_this_interaction(self):
        self.complete_interaction()
        self.clear_visualization()
        if self.in_local_mode:
            self.local_interaction = None
            self.local_interactions['interact'] = self.local_interactions['interact'][:1]
        else:
            self.interaction = None
            self.this_frame_interactions = []
        self.undo_button.setDisabled(True)
        if self.fbrs_controller is not None:
            self.fbrs_controller.unanchor()

    def set_viz_mode(self):
        self.viz_mode = self.combo.currentText()
        self.show_current_frame()

    def tl_slide(self):
        if self.waiting_to_start:
            self.waiting_to_start = False
            self.algo_timer.start()
            self.user_timer.start()
            self.console_push_text('Timers started.')

        self.reset_this_interaction()
        self.cursur = self.tl_slider.value()
        self.show_current_frame()

    def brush_slide(self):
        self.brush_size = self.brush_slider.value()
        self.brush_label.setText('Brush size: %d' % self.brush_size)
        try:
            if type(self.interaction) == FreeInteraction:
                self.interaction.set_size(self.brush_size)
        except AttributeError:
            # Initialization, forget about it
            pass

    def progress_step_cb(self):
        self.progress_num += 1
        ratio = self.progress_num/self.progress_max
        self.progress.setValue(int(ratio*100))
        self.progress.setFormat('%2.1f%%' % (ratio*100))
        QApplication.processEvents()

    def progress_total_cb(self, total):
        self.progress_max = total
        self.progress_num = -1
        self.progress_step_cb()

    def on_run(self):
        self.user_timer.pause()
        if self.interacted_mask is None:
            self.console_push_text('Cannot propagate! No interacted mask!')
            return

        self.console_push_text('Propagation started.')
        # self.interacted_mask = torch.softmax(self.interacted_mask*1000, dim=0)
        self.current_mask = self.processor.interact(self.interacted_mask, self.cursur, 
                            self.progress_total_cb, self.progress_step_cb)
        self.interacted_mask = None
        # clear scribble and reset
        self.show_current_frame()
        self.reset_this_interaction()
        self.progress.setFormat('Idle')
        self.progress.setValue(0)
        self.console_push_text('Propagation finished!')
        self.user_timer.start()

    def on_commit(self):
        self.complete_interaction()
        self.update_interacted_mask()

    def on_prev(self):
        # self.tl_slide will trigger on setValue
        self.cursur = max(0, self.cursur-1)
        self.tl_slider.setValue(self.cursur)

    def on_next(self):
        # self.tl_slide will trigger on setValue
        self.cursur = min(self.cursur+1, self.num_frames-1)
        self.tl_slider.setValue(self.cursur)

    def on_time(self):
        self.cursur += 1
        if self.cursur > self.num_frames-1:
            self.cursur = 0
        self.tl_slider.setValue(self.cursur)

    def on_play(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start(1000 / 25)

    def on_undo(self):
        if self.in_local_mode:
            if self.local_interaction is None:
                if len(self.local_interactions['interact']) > 1:
                    self.local_interactions['interact'] = self.local_interactions['interact'][:-1]
                else:
                    self.reset_this_interaction()
                self.local_interacted_mask = self.local_interactions['interact'][-1].predict()
            else:
                if self.local_interaction.can_undo():
                    self.local_interacted_mask = self.local_interaction.undo()
                else:
                    if len(self.local_interactions['interact']) > 1:
                        self.local_interaction = None
                    else:
                        self.reset_this_interaction()
                    self.local_interacted_mask = self.local_interactions['interact'][-1].predict()

            # Update visualization
            if len(self.local_vis_hist) > 0:
                # Might be empty if we are undoing the entire interaction
                self.local_vis_map, self.local_vis_alpha = self.local_vis_hist.pop()
        else:
            if self.interaction is None:
                if len(self.this_frame_interactions) > 1:
                    self.this_frame_interactions = self.this_frame_interactions[:-1]
                    self.interacted_mask = self.this_frame_interactions[-1].predict()
                else:
                    self.reset_this_interaction()
                    self.interacted_mask = self.processor.prob[:, self.cursur].clone()
            else:
                if self.interaction.can_undo():
                    self.interacted_mask = self.interaction.undo()
                else:
                    if len(self.this_frame_interactions) > 0:
                        self.interaction = None
                        self.interacted_mask = self.this_frame_interactions[-1].predict()
                    else:
                        self.reset_this_interaction()
                        self.interacted_mask = self.processor.prob[:, self.cursur].clone()

            # Update visualization
            if len(self.vis_hist) > 0:
                # Might be empty if we are undoing the entire interaction
                self.vis_map, self.vis_alpha = self.vis_hist.pop()

        # Commit changes
        self.update_interacted_mask()

    def on_reset(self):
        # DO not edit prob -- we still need the mask diff
        self.processor.masks[self.cursur].zero_()
        self.processor.np_masks[self.cursur].fill(0)
        self.current_mask[self.cursur].fill(0)
        self.reset_this_interaction()
        self.show_current_frame()

    def on_zoom_plus(self):
        self.zoom_pixels -= 25
        self.zoom_pixels = max(50, self.zoom_pixels)
        self.update_minimap()

    def on_zoom_minus(self):
        self.zoom_pixels += 25
        self.zoom_pixels = min(self.zoom_pixels, 300)
        self.update_minimap()

    def set_navi_enable(self, boolean):
        self.zoom_p_button.setEnabled(boolean)
        self.zoom_m_button.setEnabled(boolean)
        self.run_button.setEnabled(boolean)
        self.tl_slider.setEnabled(boolean)
        self.play_button.setEnabled(boolean)
        self.lcd.setEnabled(boolean)

    def on_finish_local(self):
        self.complete_interaction()
        self.finish_local_button.setDisabled(True)
        self.in_local_mode = False
        self.set_navi_enable(True)

        # Push the combined local interactions as a global interaction
        if len(self.this_frame_interactions) > 0:
            prev_soft_mask = self.this_frame_interactions[-1].out_prob
        else:
            prev_soft_mask = self.processor.prob[1:, self.cursur]
        image = self.processor.images[:,self.cursur]

        self.interaction = LocalInteraction(
            image, prev_soft_mask, (self.height, self.width), self.local_bb, 
            self.local_interactions['interact'][-1].out_prob, 
            self.processor.pad, self.local_pad
        )
        self.interaction.storage = self.local_interactions
        self.interacted_mask = self.interaction.predict()
        self.complete_interaction()
        self.update_interacted_mask()
        self.show_current_frame()

        self.console_push_text('Finished local control.')

    def hit_number_key(self, number):
        if number == self.current_object:
            return
        self.current_object = number
        if self.fbrs_controller is not None:
            self.fbrs_controller.unanchor()
        self.console_push_text('Current object changed to %d!' % number)
        self.clear_brush()
        self.vis_brush(self.last_ex, self.last_ey)
        self.update_interact_vis()
        self.show_current_frame()

    def clear_brush(self):
        self.brush_vis_map.fill(0)
        self.brush_vis_alpha.fill(0)
        if self.local_brush_vis_map is not None:
            self.local_brush_vis_map.fill(0)
            self.local_brush_vis_alpha.fill(0)

    def vis_brush(self, ex, ey):
        if self.ctrl_key:
            # Visualize the control region
            lx = int(round(min(self.local_start[0], ex)))
            ux = int(round(max(self.local_start[0], ex)))
            ly = int(round(min(self.local_start[1], ey)))
            uy = int(round(max(self.local_start[1], ey)))
            self.brush_vis_map = cv2.rectangle(self.brush_vis_map, (lx, ly), (ux, uy), 
                        (128,255,128), thickness=-1)
            self.brush_vis_alpha = cv2.rectangle(self.brush_vis_alpha, (lx, ly), (ux, uy), 
                        0.5, thickness=-1)
        else:
            # Visualize the brush (yeah I know)
            if self.in_local_mode:
                self.local_brush_vis_map = cv2.circle(self.local_brush_vis_map, 
                        (int(round(ex)), int(round(ey))), self.brush_size//2+1, color_map[self.current_object], thickness=-1)
                self.local_brush_vis_alpha = cv2.circle(self.local_brush_vis_alpha, 
                        (int(round(ex)), int(round(ey))), self.brush_size//2+1, 0.5, thickness=-1)
            else:
                self.brush_vis_map = cv2.circle(self.brush_vis_map, 
                        (int(round(ex)), int(round(ey))), self.brush_size//2+1, color_map[self.current_object], thickness=-1)
                self.brush_vis_alpha = cv2.circle(self.brush_vis_alpha, 
                        (int(round(ex)), int(round(ey))), self.brush_size//2+1, 0.5, thickness=-1)

    def enter_local_control(self):
        self.in_local_mode = True
        lx = int(round(min(self.local_start[0], self.local_end[0])))
        ux = int(round(max(self.local_start[0], self.local_end[0])))
        ly = int(round(min(self.local_start[1], self.local_end[1])))
        uy = int(round(max(self.local_start[1], self.local_end[1])))

        # Reset variables
        self.local_bb = (lx, ux, ly, uy)
        self.local_interactions = {}
        self.local_interactions['interact'] = []
        self.local_interaction = None

        # Initial info
        if len(self.this_local_interactions) == 0:
            prev_soft_mask = self.processor.prob[1:, self.cursur]
        else:
            prev_soft_mask = self.this_local_interactions[-1].out_prob
        self.local_interactions['bounding_box'] = self.local_bb
        self.local_interactions['cursur'] = self.cursur
        init_interaction = CropperInteraction(self.processor.images[:,self.cursur], 
                                    prev_soft_mask, self.processor.pad, self.local_bb)
        self.local_interactions['interact'].append(init_interaction)

        self.local_interacted_mask = init_interaction.out_mask
        self.local_torch_im = init_interaction.im_crop
        self.local_np_im = self.images[self.cursur][ly:uy+1, lx:ux+1, :]
        self.local_pad = init_interaction.pad

        # initialize the local visualization maps
        h, w = init_interaction.h, init_interaction.w
        self.local_vis_map = np.zeros((h, w, 3), dtype=np.uint8)
        self.local_vis_alpha = np.zeros((h, w, 1), dtype=np.float32)
        self.local_brush_vis_map = np.zeros((h, w, 3), dtype=np.uint8)
        self.local_brush_vis_alpha = np.zeros((h, w, 1), dtype=np.float32)
        self.local_vis_hist = deque(maxlen=100)
        self.local_height, self.local_width = h, w

        # Refresh self.viz
        self.minimap_in_local_drawn = False
        self.minimap_in_local = self.viz_with_stroke
        self.update_interacted_mask()
        self.finish_local_button.setEnabled(True)
        self.undo_button.setEnabled(False)
        self.set_navi_enable(False)

        self.console_push_text('Entered local control.')

    def on_press(self, event):
        if self.waiting_to_start:
            self.waiting_to_start = False
            self.algo_timer.start()
            self.user_timer.start()
            self.console_push_text('Timers started.')

        self.user_timer.pause()
        ex, ey = self.get_scaled_pos(event.x(), event.y())
        # Check for ctrl key
        modifiers = QApplication.keyboardModifiers()
        if not self.in_local_mode and modifiers == QtCore.Qt.ControlModifier:
            # Start specifying the local mode
            self.ctrl_key = True
        else:
            self.ctrl_key = False

        self.pressed = True
        self.right_click = (event.button() != 1)
        # Push last vis map into history
        if self.in_local_mode:
            self.local_vis_hist.append((self.local_vis_map.copy(), self.local_vis_alpha.copy()))
        else:
            self.vis_hist.append((self.vis_map.copy(), self.vis_alpha.copy()))
        if self.ctrl_key:
            # Wrap up the last interaction
            self.complete_interaction()
            # Labeling a local control field
            self.local_start = ex, ey
        else:
            # Ordinary interaction (might be in local mode)
            if self.in_local_mode:
                if self.local_interaction is None:
                    prev_soft_mask = self.local_interactions['interact'][-1].out_prob
                else:
                    prev_soft_mask = self.local_interaction.out_prob
                prev_hard_mask = self.local_max_mask
                image = self.local_torch_im
                h, w = self.local_height, self.local_width
            else:
                if self.interaction is None:
                    if len(self.this_frame_interactions) > 0:
                        prev_soft_mask = self.this_frame_interactions[-1].out_prob
                    else:
                        prev_soft_mask = self.processor.prob[1:, self.cursur]
                else:
                    # Not used if the previous interaction is still valid
                    # Don't worry about stacking effects here
                    prev_soft_mask = self.interaction.out_prob
                prev_hard_mask = self.processor.masks[self.cursur]
                image = self.processor.images[:,self.cursur]
                h, w = self.height, self.width

            last_interaction = self.local_interaction if self.in_local_mode else self.interaction
            new_interaction = None
            if self.curr_interaction == 'Scribble':
                if last_interaction is None or type(last_interaction) != ScribbleInteraction:
                    self.complete_interaction()
                    new_interaction = ScribbleInteraction(image, prev_hard_mask, (h, w), 
                                self.s2m_controller, self.num_objects)
            elif self.curr_interaction == 'Free':
                if last_interaction is None or type(last_interaction) != FreeInteraction:
                    self.complete_interaction()
                    if self.in_local_mode:
                        new_interaction = FreeInteraction(image, prev_soft_mask, (h, w), 
                                self.num_objects, self.local_pad)
                    else:
                        new_interaction = FreeInteraction(image, prev_soft_mask, (h, w), 
                                self.num_objects, self.processor.pad)
                    new_interaction.set_size(self.brush_size)
            elif self.curr_interaction == 'Click':
                if (last_interaction is None or type(last_interaction) != ClickInteraction 
                        or last_interaction.tar_obj != self.current_object):
                    self.complete_interaction()
                    self.fbrs_controller.unanchor()
                    new_interaction = ClickInteraction(image, prev_soft_mask, (h, w), 
                                self.fbrs_controller, self.current_object, self.processor.pad)

            if new_interaction is not None:
                if self.in_local_mode:
                    self.local_interaction = new_interaction
                else:
                    self.interaction = new_interaction

        # Just motion it as the first step
        self.on_motion(event)
        self.user_timer.start()

    def on_motion(self, event):
        ex, ey = self.get_scaled_pos(event.x(), event.y())
        self.last_ex, self.last_ey = ex, ey
        self.clear_brush()
        # Visualize
        self.vis_brush(ex, ey)
        if self.pressed:
            if not self.ctrl_key:
                if self.curr_interaction == 'Scribble' or self.curr_interaction == 'Free':
                    obj = 0 if self.right_click else self.current_object
                    # Actually draw it if dragging
                    if self.in_local_mode:
                        self.local_vis_map, self.local_vis_alpha = self.local_interaction.push_point(
                            ex, ey, obj, (self.local_vis_map, self.local_vis_alpha)
                        )
                    else:
                        self.vis_map, self.vis_alpha = self.interaction.push_point(
                            ex, ey, obj, (self.vis_map, self.vis_alpha)
                        )
        self.update_interact_vis()
        self.update_minimap()

    def update_interacted_mask(self):
        if self.in_local_mode:
            self.local_max_mask = torch.argmax(self.local_interacted_mask, 0)
            max_mask = unpad_3dim(self.local_max_mask, self.local_pad)
            self.local_np_mask = (max_mask.detach().cpu().numpy()[0]).astype(np.uint8)
        else:
            self.processor.update_mask_only(self.interacted_mask, self.cursur)
            self.current_mask[self.cursur] = self.processor.np_masks[self.cursur]
        self.show_current_frame()

    def complete_interaction(self):
        if self.in_local_mode:
            if self.local_interaction is not None:
                self.clear_visualization()
                self.local_interactions['interact'].append(self.local_interaction)
                self.local_interaction = None
                self.undo_button.setDisabled(False)
        else:
            if self.interaction is not None:
                self.clear_visualization()
                self.interactions['annotated_frame'].append(self.cursur)
                self.interactions['interact'][self.cursur].append(self.interaction)
                self.this_frame_interactions.append(self.interaction)
                self.interaction = None
                self.undo_button.setDisabled(False)

    def on_release(self, event):
        self.user_timer.pause()
        ex, ey = self.get_scaled_pos(event.x(), event.y())
        if self.ctrl_key:
            # Enter local control mode
            self.clear_visualization()
            self.local_end = ex, ey
            self.enter_local_control()
        else:
            self.console_push_text('Interaction %s at frame %d.' % (self.curr_interaction, self.cursur))
            # Ordinary interaction (might be in local mode)
            if self.in_local_mode:
                interaction = self.local_interaction
            else:
                interaction = self.interaction

            if self.curr_interaction == 'Scribble' or self.curr_interaction == 'Free':
                self.on_motion(event)
                interaction.end_path()
                if self.curr_interaction == 'Free':
                    self.clear_visualization()
            elif self.curr_interaction == 'Click':
                ex, ey = self.get_scaled_pos(event.x(), event.y())
                if self.in_local_mode:
                    self.local_vis_map, self.local_vis_alpha = interaction.push_point(ex, ey,
                        self.right_click, (self.local_vis_map, self.local_vis_alpha))
                else:
                    self.vis_map, self.vis_alpha = interaction.push_point(ex, ey,
                        self.right_click, (self.vis_map, self.vis_alpha))

            if self.in_local_mode:
                self.local_interacted_mask = interaction.predict()
            else:
                self.interacted_mask = interaction.predict()
            self.update_interacted_mask()

        self.pressed = self.ctrl_key = self.right_click = False
        self.undo_button.setDisabled(False)
        self.user_timer.start()

    def debug_pressed(self):
        self.debug_mask, self.interacted_mask = self.interacted_mask, self.debug_mask

        self.processor.update_mask_only(self.interacted_mask, self.cursur)
        self.current_mask[self.cursur] = self.processor.np_masks[self.cursur]
        self.show_current_frame()

    def wheelEvent(self, event):
        ex, ey = self.get_scaled_pos(event.x(), event.y())
        if self.curr_interaction == 'Free':
            self.brush_slider.setValue(self.brush_slider.value() + event.angleDelta().y()//30)
        self.clear_brush()
        self.vis_brush(ex, ey)
        self.update_interact_vis()
        self.update_minimap()


def seg_video(img_path=None, video_path=None, MIVOS_PATH='/home/yihua/nips2022/code/repos/MiVOS/'):
    
    # Arguments parsing
    parser = ArgumentParser()
    parser.add_argument('--prop_model', default=MIVOS_PATH+'saves/propagation_model.pth')
    parser.add_argument('--fusion_model', default=MIVOS_PATH+'saves/fusion.pth')
    parser.add_argument('--s2m_model', default=MIVOS_PATH+'saves/s2m.pth')
    parser.add_argument('--fbrs_model', default=MIVOS_PATH+'saves/fbrs.pth')
    parser.add_argument('--images', help='Folders containing input images. Either this or --video need to be specified.')
    parser.add_argument('--video', help='Video file readable by OpenCV. Either this or --images need to be specified.', default='example/example.mp4')
    parser.add_argument('--num_objects', help='Default: 1 if no masks provided, masks.max() otherwise', type=int)
    parser.add_argument('--mem_freq', default=5, type=int)
    parser.add_argument('--mem_profile', default=0, type=int, help='0 - Faster and more memory intensive; 2 - Slower and less memory intensive. Default: 0.')
    parser.add_argument('--masks', help='Optional, Ground truth masks', default=None)
    parser.add_argument('--no_amp', help='Turn off AMP', action='store_true')
    parser.add_argument('--resolution', help='Pass -1 to use original size', default=480, type=int)
    args = parser.parse_args()

    with torch.cuda.amp.autocast(enabled=not args.no_amp):
        # Load our checkpoint
        prop_saved = torch.load(args.prop_model)
        prop_model = PropagationNetwork().cuda().eval()
        prop_model.load_state_dict(prop_saved)

        fusion_saved = torch.load(args.fusion_model)
        fusion_model = FusionNet().cuda().eval()
        fusion_model.load_state_dict(fusion_saved)

        # Loads the S2M model
        if args.s2m_model is not None:
            s2m_saved = torch.load(args.s2m_model)
            s2m_model = S2M().cuda().eval()
            s2m_model.load_state_dict(s2m_saved)
        else:
            s2m_model = None

        # Loads the images/masks
        args.images = img_path
        args.video = video_path
        sv_path = img_path if video_path is None else video_path
        if sv_path.endswith('/'):
            sv_path = sv_path[:-1]
        sv_path = '/'.join(sv_path.split('/')[:-1])
        if os.path.exists(sv_path + '/mask'):
        	return sv_path + '/mask'
        if args.images is not None:
            images = load_images(args.images, args.resolution if args.resolution > 0 else None)
        elif args.video is not None:
            images = load_video(args.video, args.resolution if args.resolution > 0 else None)
        else:
            raise NotImplementedError('You must specify either --images or --video!')

        if args.masks is not None:
            masks = load_masks(args.masks)
        else:
            masks = None

        # Determine the number of objects
        num_objects = args.num_objects
        if num_objects is None:
            if masks is not None:
                num_objects = masks.max()
            else:
                num_objects = 1

        s2m_controller = S2MController(s2m_model, num_objects, ignore_class=255)
        if args.fbrs_model is not None:
            fbrs_controller = FBRSController(args.fbrs_model)
        else:
            fbrs_controller = None

        app = QApplication(sys.argv)
        ex = App(prop_model, fusion_model, s2m_controller, fbrs_controller, 
                    images, masks, num_objects, args.mem_freq, args.mem_profile, sv_path=sv_path)
        # sys.exit(app.exec_())
        app.exec_()
        return ex.sv_path + '/mask'

