# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 11:32:10 2024

@author: lh
"""

import tkinter as tk
from PIL import Image, ImageTk
import os, shutil
import cv2
import tempfile
import numpy as np
from subprocess import call
import argparse
os.environ['PYOPENGL_PLATFORM'] = 'osmesa' #egl
import pyrender
import trimesh
from psbody.mesh import Mesh

from tkinter import ttk
import time

au_dic = {'happy':[4,9],'sad':[0,2,11],'surprise':[0,1,3,18],'fear':[0,1,2,3,5,15,20],'angry':[2,3,6,17],'disgust':[6,11,12],'contempt':[2,10,31]}

emotion_positions = {
                    'happy':      (0.8,  0.8),   # 高愉悦高激活
                    'sad':        (-0.8, -0.5),  # 低愉悦低激活
                    'angry':      (1,  -0.7),  # 低愉悦高激活
                    'fear':       (-0.4,  0.9),
                    'surprise':   (0.6,   1.0),
                    'disgust':    (-0.9,  0.3),
                    'neutral':    (0.0,   0.0)
                }

def compute_emotion_weights(x, y):
    # 距离越小权重越大，这里用高斯分布
    sigma = 0.1  # 控制影响范围
    weights = {}
    for emo, (ex, ey) in emotion_positions.items():
        dist = np.linalg.norm([x - ex, y - ey])
        weight = np.exp(-dist**2 / (2 * sigma**2))
        weights[emo] = weight
    # 归一化为概率
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    return weights

def on_motion(event):
    x, y = event.x, event.y
    canvas.delete("dot")
    r = 3
    canvas.create_oval(x - r, y - r, x + r, y + r, fill='red', tag="dot")
    update_image(x, y, mode="emotion")

def update_time():
    now = time.strftime('%Y-%m-%d %H:%M:%S')  # 格式化当前时间
    label_g_time.config(text=now)                     # 更新标签文本
    root.after(1000, update_time)  

# The implementation of rendering is borrowed from VOCA: https://github.com/TimoBolkart/voca/blob/master/utils/rendering.py
def render_mesh_helper(mesh, t_center, rot=np.zeros(3), tex_img=None, z_offset=0):
    
    camera_params = {'c': np.array([400, 400]),
                         'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                         'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}

    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

    mesh_copy = Mesh(mesh.v, mesh.f)
    mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v-t_center).T).T+t_center
    
    intensity = 2.0

    primitive_material = pyrender.material.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor=[0.3, 0.3, 0.3, 1.0],
                metallicFactor=0.8, 
                roughnessFactor=0.8 
            )


    tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f)

    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material,smooth=True)


    scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])
    
    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                      fy=camera_params['f'][1],
                                      cx=camera_params['c'][0],
                                      cy=camera_params['c'][1],
                                      znear=frustum['near'],
                                      zfar=frustum['far'])

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3,3] = np.array([0, 0, 1.0-z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])

    angle = np.pi / 6.0
    pos = camera_pose[:3,3]
    light_color = np.array([1., 1., 1.])
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3,3] = pos
    scene.add(light, pose=light_pose.copy())
    
    light_pose[:3,3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] =  cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
        color, _ = r.render(scene, flags=flags)
    except:
        print('pyrender: Failed rendering frame')

        color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]

def render_sequence_meshes( sequence_vertices, template,):
    num_frames = sequence_vertices.shape[0]
    center = np.mean(sequence_vertices[0], axis=0)
    for i_frame in range(num_frames):
        render_mesh = Mesh(sequence_vertices[i_frame], template.f)
        pred_img = render_mesh_helper(render_mesh, center, tex_img=None)
        pred_img = pred_img.astype(np.uint8)
    return pred_img
    
def update_image(x=None,y=None,mode="slider"):
    role_index = role.get()
    if role_index == 0 :
        vertices = np.load(('/home/lh/lihao/AU/Aublendnet-main/dataset/render/npy/6/6.npy')).reshape(-1,5023,3)[1:]
        temp = np.load('/home/lh/lihao/AU/Aublendnet-main/dataset/render/npy/6/6.npy')[0].reshape(5023,3)
        diff = vertices - temp
    

    if x is not None and y is not None:
        start_time = time.time()
        happy = 0
        for index in au_dic['happy']:
            happy = happy + diff[index:index+1] 
        sad = 0
        for index in au_dic['sad']:
             sad =  sad + diff[index:index+1] 
        angry= 0
        for index in au_dic['angry']:
             angry =  angry + diff[index:index+1] 
        fear= 0
        for index in au_dic['fear']:
             fear =  fear + diff[index:index+1]
        surprise= 0
        for index in au_dic['surprise']:
             surprise =  surprise + diff[index:index+1] 

        disgust= 0
        for index in au_dic['disgust']:
            disgust =  disgust + diff[index:index+1] 
        #start_time = time.time()
        weights = compute_emotion_weights(x/250, y/200)
        print(weights)
        vertices_out = temp + weights['happy']*happy + weights['sad']*sad+ weights['angry']*angry+ weights['fear']*fear+ weights['surprise']* surprise+ weights['disgust']* disgust
        end_time = time.time()
        elapsed = 1/(end_time - start_time )
        label_time.config(text=f"FPS:{elapsed:.0f}") 
        image_path = '/home/lh/lihao/AU/Aublendnet-main/img1.png'
    else:
        start_time = time.time()
        image_path = '/home/lh/lihao/AU/Aublendnet-main/img1.png'
        vertices_out = temp
        index_dif = 0
        for i in range(4):
            for j in range(8):
                cont = sliders[i][j].get()
                vertices_out = vertices_out + cont*diff[index_dif:index_dif+1]/100
                index_dif = index_dif+1
        end_time = time.time()
        elapsed = 1/(end_time - start_time )
        label_time.config(text=f"FPS:{elapsed:.0f}") 

    #out_path = os.path.join('/home/lh/lihao/AU/Aublendnet-main','dif_blendshape.npy')
    #np.save(out_path,vertices_out-temp.reshape(-1,5023,3) )
    img=render_sequence_meshes(vertices_out, template)
    cv2.imwrite(image_path,img)
        
    image = Image.open(image_path)
    image = image.resize((250, 250))  # 调整图片大小
    photo = ImageTk.PhotoImage(image)
        
    img_lable.config(image=photo)
    img_lable.image=photo

def auto_update_image():
    update_image() # 调用更新图片的函数
    root.after(33, auto_update_image) # 每33毫秒再次调用，相当于每秒30次

def reset_sliders(): 
    for frame_sliders in sliders: 
         for slider in frame_sliders: 
            slider.set(0) # 将滑动条的值设置为0 update_image() # 重置后更新图片
    update_image()

lable = ['AU1 Inner brow raiser','AU2 Out brow raiser','AU4 Brow lowerer','AU5 Upper lid raiser','AU6 Cheek raiser','AU7 Lid tightener','AU9 Nose Wrinkler','AU10 Upper lip raiser','AU11 Nasolabial deepener','AU12 Lip corner puller','AU14 Dimpler','AU15 Lip corner depressor','AU16 lower lip depressor','AU17 Chin raiser','AU18 Lip Pucker','AU20 Lip stretcher','AU22 Lip Funneler','AU23 Lip Tightener','AU24 Lip pressor','AU25 Lip part','AU26 Jaw Drop','AU27 Mouth stretch','AU28 Lip Suck','AU29 Jaw thrust','AU30R Jaw sideways (left)','AU30R Jaw sideways (right)','AU33 Cheek blow','AU45 Blink','AU61 Eyes turn left','AU62 Eyes turn right','AU63 Eyes up','AU64 Eyes down']
template_file = '/home/lh/lihao/AU/Aublendnet-main/dataset/FLAME_sample.ply'
template = Mesh(filename=template_file)

# 创建主窗口
root = tk.Tk()
root.title("控制面板")

frames =[]
sliders = []
lable_index = 0
for group in range(4):
    frame = tk.Frame(root,borderwidth=2,relief='groove')
    frame.grid(row=0,column=group,padx=10,pady=10)
    frames.append(frame)
    frame_sliders = []
    for j in range(8):
        slider = tk.Scale(frame, from_=0,to =100 ,orient='horizontal',label=lable[lable_index],command=lambda _: update_image(mode="slider"),length=200)
        lable_index = lable_index + 1
        slider.grid(row=j,column=0,padx=5,pady=5)
        frame_sliders.append(slider)
    sliders.append(frame_sliders)

role = tk.Scale(root, from_=0,to =5 ,orient='horizontal',label='Role',command=lambda _:update_image)
role.grid(row=2,column=3,columnspan=2,pady=0)

# 按钮点击显示当前图片
img_lable = tk.Label(root)
img_lable.grid(row=1, column=0,columnspan=4,padx=(30, 0),pady=10 )

#img_lable1 = tk.Label(root)
#img_lable1.grid(row=2, column=0,columnspan=4, pady=00)

button = tk.Button(root, text="显示图片", command=update_image)
button.grid(row=1,column=3,columnspan=2,pady=0)

reset_button = tk.Button(root, text="Reset Sliders", command=reset_sliders) 
reset_button.grid(row=1, column=3, columnspan=2, pady=6)

label_time = tk.Label(root, font=("Times New Roman", 24), fg="black")
label_time.grid(row=2, column=0)

label_g_time = tk.Label(root, font=('Times New Roman', 24))
label_g_time.grid(row=1, column=0)

canvas = tk.Canvas(root, width=250, height=200, bg='white')
canvas.grid(row=2, column=1,columnspan=2)

canvas.bind("<Motion>", on_motion) 
# 初始化Canvas内容
update_image()
#auto_update_image()

update_time() 
# 启动主循环
root.mainloop()