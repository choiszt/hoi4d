import numpy as np
import glob
from PIL import Image,ImageDraw,ImageFont
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import cv2
from matplotlib.colors import ListedColormap, BoundaryNorm
import random
from scipy import ndimage
import json
import warnings
import sys;sys.path.append("/mnt/ve_share/liushuai/panoptic-segment-anything/hoi4d/prepare_4Dseg")
from palette import pal_color_map
from utils import category2label_map, category2label_map_instanceseg
palette=pal_color_map()
import tqdm
warnings.filterwarnings("ignore", category=DeprecationWarning)
class hoimaskpinner:
	def __init__(self):
		self.font = ImageFont.truetype("/mnt/ve_share/liushuai/panoptic-segment-anything/OpenSans-Bold.ttf",20)
		self.tickfont = ImageFont.truetype("/mnt/ve_share/liushuai/panoptic-segment-anything/OpenSans-Bold.ttf",40)
		with open("/mnt/ve_share/liushuai/panoptic-segment-anything/hoi4d/categories.txt")as f:
			t=f.readlines()
		self.categories={"47":"hands"} #label文件中没有，但utils中手对应的mapping是47
		for i in range(43):
			self.categories[t[i+43].strip()]=t[i].strip()
		self.palette=pal_color_map()
		self.font = ImageFont.truetype("/mnt/ve_share/liushuai/panoptic-segment-anything/OpenSans-Bold.ttf",20)
		self.tickfont = ImageFont.truetype("/mnt/ve_share/liushuai/panoptic-segment-anything/OpenSans-Bold.ttf",40)
		self.category2label_map=category2label_map
		self.category2label_map_instanceseg=category2label_map_instanceseg
	def get_mask_and_label(self,path):
		l = path.split('/')
		for i in l:
			if 'C' in i:
				category = i
		return self.f(path, category)


	def f(self,img_path, category):
		image = cv2.imread(img_path)[:, :, ::-1]
		assert image.shape == (1080, 1920, 3)
		# image = shift_mask(image, img_path)
		color_map = pal_color_map() #palette

		arrs = []
		labels = []  # semantic segmentation
		labels_instanceseg = []  # instance segmentation
		for i in range(10):
			color = color_map[i + 1]
			valid = (image[..., 0] == color[0]) & (image[..., 1] == color[1]) & (image[..., 2] == color[2])
			if np.sum(valid) == 0:
				continue
			if i >= len(category2label_map[category]): #大于类数肯定不对
				continue
			arrs.append(valid)
			labels.append(category2label_map[category][i])
			labels_instanceseg.append(category2label_map_instanceseg[category][i])
		
		return arrs, np.array(labels), np.array(labels_instanceseg)
	def getindex(self,list,catelist):#输入一个list 返回一个字典 包括这个key 他的类
		dict={}
		for i in range(len(list)):
			if(list[i] not in dict.keys()):
				dict[list[i]]=[[i],self.categories[str(catelist[i])]]#记录他的index 且相同的label的category都相同，就第一次记录就可以了
			else:
				dict[list[i]][0].append(i)
		return dict
	def drawmask(self,frame,panopticmask,colordict,ele):
		pinnedmask=panopticmask
		unique_values=np.unique(panopticmask)
		expanded_mask=np.repeat(panopticmask[:, :, np.newaxis], 3, axis=2)
		
		for value in unique_values:
			expanded_mask[:,:,0][panopticmask == value] = colordict[value][0][0] #把颜色附在mask上
			expanded_mask[:,:,1][panopticmask == value] = colordict[value][0][1] 
			expanded_mask[:,:,2][panopticmask == value] = colordict[value][0][2] 
		expanded_mask = expanded_mask.astype(np.uint8)
		maskimage=Image.fromarray(expanded_mask)
		draw = ImageDraw.Draw(maskimage)
		for key in colordict.keys():
			if np.count_nonzero(pinnedmask == key) != 0: #防止重心为0出现Nan
				cy, cx = ndimage.center_of_mass(pinnedmask == key)
				if colordict[key][1] is not None:
					text_width, text_height = draw.textsize (f'{key}_{colordict[key][1]}', font=self.font)
					draw.rectangle([cx, cy,cx+text_width, cy+text_height],fill=(0,0,0))
					draw.text((cx, cy), f'{key}_{colordict[key][1]}', fill=tuple(colordict[key][0]), font=self.font)
				elif key==0:
					pass
				else:
					text_width, text_height = draw.textsize(f'{key}', font=self.font)
					draw.rectangle([cx, cy,cx+text_width, cy+text_height],fill=(0,0,0))
					draw.text((cx, cy), f'{key}', fill=tuple(colordict[key][0]), font=self.font)
		
		text_width, text_height = draw.textsize(f"{ele.strip('.png')[-4:]}", font=self.tickfont)
		draw.rectangle([0, 0,text_width, text_height],fill=(0,0,0))
		draw.text((0, 0), f"{ele.strip('.png')[-4:]}", fill=(255,255,255), font=self.tickfont)
		image_array = np.array(frame)
		expanded_mask=np.array(maskimage)
		alpha=0.4
		foreground = (image_array * alpha + np.ones(image_array.shape) * (1 - alpha) * expanded_mask)
		image = foreground.astype('uint8')
		token=ele.strip(".png").split("/")[-1][-4:]
		cv2.imwrite(f"/mnt/ve_share/liushuai/panoptic-segment-anything/hoi4d/results/{token}.jpg",image)
		return image




class maskpinner:
	def __init__(self):
		with open('/mnt/ve_share/liushuai/SegmentAnyRGBD-main/annotation/train_24.json', 'r') as f:
			self.data= json.load(f)
		with open('/mnt/ve_share/liushuai/SegmentAnyRGBD-main/annotation/val_24.json', 'r') as f:
			self.valdata = json.load(f)
		
		self.trainlist,self.testlist=self._createlist(self.data,self.valdata)
		self._palette = [
    255, 255, 255, 0, 0, 139, 255, 255, 84, 0, 255, 0, 139, 0, 139, 0, 128, 128,
    128, 128, 128, 139, 0, 0, 218, 165, 32, 144, 238, 144, 160, 82, 45, 148, 0,
    211, 255, 0, 255, 30, 144, 255, 255, 218, 185, 85, 107, 47, 255, 140, 0,
    50, 205, 50, 123, 104, 238, 240, 230, 140, 72, 61, 139, 128, 128, 0, 0, 0,
    205, 221, 160, 221, 143, 188, 143, 127, 255, 212, 176, 224, 230, 244, 164,
    96, 250, 128, 114, 70, 130, 180, 0, 128, 0, 173, 255, 47, 255, 105, 180,
    238, 130, 238, 154, 205, 50, 220, 20, 60, 176, 48, 96, 0, 206, 209, 0, 191,
    255, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43, 43, 44, 44, 44, 45, 45,
    45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50, 50, 51, 51, 51,
    52, 52, 52, 53, 53, 53, 54, 54, 54, 55, 55, 55, 56, 56, 56, 57, 57, 57, 58,
    58, 58, 59, 59, 59, 60, 60, 60, 61, 61, 61, 62, 62, 62, 63, 63, 63, 64, 64,
    64, 65, 65, 65, 66, 66, 66, 67, 67, 67, 68, 68, 68, 69, 69, 69, 70, 70, 70,
    71, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 75, 75, 75, 76, 76, 76, 77,
    77, 77, 78, 78, 78, 79, 79, 79, 80, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83,
    83, 84, 84, 84, 85, 85, 85, 86, 86, 86, 87, 87, 87, 88, 88, 88, 89, 89, 89,
    90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93, 94, 94, 94, 95, 95, 95, 96,
    96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99, 100, 100, 100, 101, 101, 101,
    102, 102, 102, 103, 103, 103, 104, 104, 104, 105, 105, 105, 106, 106, 106,
    107, 107, 107, 108, 108, 108, 109, 109, 109, 110, 110, 110, 111, 111, 111,
    112, 112, 112, 113, 113, 113, 114, 114, 114, 115, 115, 115, 116, 116, 116,
    117, 117, 117, 118, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121,
    122, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126,
    127, 127, 127, 128, 128, 128, 129, 129, 129, 130, 130, 130, 131, 131, 131,
    132, 132, 132, 133, 133, 133, 134, 134, 134, 135, 135, 135, 136, 136, 136,
    137, 137, 137, 138, 138, 138, 139, 139, 139, 140, 140, 140, 141, 141, 141,
    142, 142, 142, 143, 143, 143, 144, 144, 144, 145, 145, 145, 146, 146, 146,
    147, 147, 147, 148, 148, 148, 149, 149, 149, 150, 150, 150, 151, 151, 151,
    152, 152, 152, 153, 153, 153, 154, 154, 154, 155, 155, 155, 156, 156, 156,
    157, 157, 157, 158, 158, 158, 159, 159, 159, 160, 160, 160, 161, 161, 161,
    162, 162, 162, 163, 163, 163, 164, 164, 164, 165, 165, 165, 166, 166, 166,
    167, 167, 167, 168, 168, 168, 169, 169, 169, 170, 170, 170, 171, 171, 171,
    172, 172, 172, 173, 173, 173, 174, 174, 174, 175, 175, 175, 176, 176, 176,
    177, 177, 177, 178, 178, 178, 179, 179, 179, 180, 180, 180, 181, 181, 181,
    182, 182, 182, 183, 183, 183, 184, 184, 184, 185, 185, 185, 186, 186, 186,
    187, 187, 187, 188, 188, 188, 189, 189, 189, 190, 190, 190, 191, 191, 191,
    192, 192, 192, 193, 193, 193, 194, 194, 194, 195, 195, 195, 196, 196, 196,
    197, 197, 197, 198, 198, 198, 199, 199, 199, 200, 200, 200, 201, 201, 201,
    202, 202, 202, 203, 203, 203, 204, 204, 204, 205, 205, 205, 206, 206, 206,
    207, 207, 207, 208, 208, 208, 209, 209, 209, 210, 210, 210, 211, 211, 211,
    212, 212, 212, 213, 213, 213, 214, 214, 214, 215, 215, 215, 216, 216, 216,
    217, 217, 217, 218, 218, 218, 219, 219, 219, 220, 220, 220, 221, 221, 221,
    222, 222, 222, 223, 223, 223, 224, 224, 224, 225, 225, 225, 226, 226, 226,
    227, 227, 227, 228, 228, 228, 229, 229, 229, 230, 230, 230, 231, 231, 231,
    232, 232, 232, 233, 233, 233, 234, 234, 234, 235, 235, 235, 236, 236, 236,
    237, 237, 237, 238, 238, 238, 239, 239, 239, 240, 240, 240, 241, 241, 241,
    242, 242, 242, 243, 243, 243, 244, 244, 244, 245, 245, 245, 246, 246, 246,
    247, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250, 250, 251, 251, 251,
    252, 252, 252, 253, 253, 253, 254, 254, 254, 255, 255, 255, 0, 0, 0
]
		self.color_palette = np.array(self._palette).reshape(-1, 3)
		self.stuffdict={5001:[[0],'floor'],5022:[[0],'ground'],5043:[[0],'wall'],5064:[[0],'window'],5085:[[0],'stair'],5106:[[0],'fence'],5127:[[0],'grass'],5148:[[0],'sky']}
		self.font = ImageFont.truetype("/mnt/ve_share/liushuai/panoptic-segment-anything/OpenSans-Bold.ttf",20)
		self.tickfont = ImageFont.truetype("/mnt/ve_share/liushuai/panoptic-segment-anything/OpenSans-Bold.ttf",40)
	def _findtag(self,maskdir,key,data,valdata): #传入mask的地址和mask的unique key
		videoname=maskdir[0].split('/')[-3]
		for i in data['annotations']:
			if i['model'].split('/')[0]==videoname:
				if(i['obj_id']==key):
					for category in data['categories']:
						if(i['category_id']==category['id']):
							return category['name']
		for i in valdata['annotations']:
			if i['model'].split('/')[0]==videoname:
				if(i['obj_id']==key):
					for category in valdata['categories']:
						if(i['category_id']==category['id']):
							return category['name']
	def _build_colordict(self,maskdir,data,valdata):#构建这个video场景的colordict
		color_dict={}
		for num in range(len(maskdir)):
			visible_mask=np.load(maskdir[num])
			for value in np.unique(visible_mask):
				if value not in color_dict.keys():
					color_dict.update({value:[random.choices(range(256)),self._findtag(maskdir,value,data,valdata)]})
		return color_dict
	def _createlist(self,data,valdata):
		trainlist=[]
		for i in self.data['annotations']:
			if i['camera'].split('/')[0] in self.targetlist:
				if i['camera'].split('/')[0] not in trainlist:
					trainlist.append(i['camera'].split('/')[0])
		testlist=[]
		for i in self.valdata['annotations']:
			if i['camera'].split('/')[0] in self.targetlist:
				if i['camera'].split('/')[0] not in testlist:
					testlist.append(i['camera'].split('/')[0])
		return trainlist,testlist
	
	def colordict(self,maskdir):
		self.colordict=self._build_colordict(maskdir,self.data,self.valdata)
		# self.colordict.update(self.stuffdict)
		self.colordict=(sorted(self.colordict.items()))
		for i in range(len(self.colordict)):
			self.colordict[i][1][0]=(self.color_palette[i%256]+77*i)%255
		self.tick=len(self.colordict)
		return self.colordict
	def updatecolordict(self,panopticmask,annotations,category_names):
		uniquepix,cnt=np.unique(panopticmask,return_counts=True)
		self.colordict=dict(self.colordict)
		for i in uniquepix:
			for anno in annotations:
				if anno['id']==i:
					self.cate=category_names[anno['category_id']]
			if i not in self.colordict.keys():
				self.colordict.update({i:[(self.color_palette[self.tick%256]+77*self.tick)%255,self.cate]})
				self.tick+=1
	def getpinnedmask(self,pred_mask_sam_depth,gt_mask_path):
		self.gt_mask=np.load(gt_mask_path)
		self.pred_mask_sam_depth=pred_mask_sam_depth
		self.pinnedmask=self.pred_mask_sam_depth.copy()
		categories=[cate['name'] for cate in self.data['categories']]
		for i in self.colordict.items():
			if i[0]>5000:
				if(i[1][1] in categories):
					self.pinnedmask[(self.pred_mask_sam_depth==i[0])]=0
		self.pinnedmask[(self.pred_mask_sam_depth==5000)]=0 #5000的地方规定为背景
		self.pinnedmask[self.gt_mask!=0]=self.gt_mask[self.gt_mask!=0]
	def drawmask(self,imagepath):
		
		pinnedmask=self.pinnedmask
		unique_values=np.unique(pinnedmask)
		expanded_mask=np.repeat(pinnedmask[:, :, np.newaxis], 3, axis=2)
		colordict=self.colordict
		
		for value in unique_values:
			expanded_mask[:,:,0][pinnedmask == value] = colordict[value][0][0] #把颜色附在mask上
			expanded_mask[:,:,1][pinnedmask == value] = colordict[value][0][1] 
			expanded_mask[:,:,2][pinnedmask == value] = colordict[value][0][2] 
		expanded_mask = expanded_mask.astype(np.uint8)
		maskimage=Image.fromarray(expanded_mask)
		draw = ImageDraw.Draw(maskimage)
		for key in colordict.keys():
			if np.count_nonzero(pinnedmask == key) != 0: #防止重心为0出现Nan
				cy, cx = ndimage.center_of_mass(pinnedmask == key)
				if colordict[key][1] is not None:
					text_width, text_height = draw.textsize (f'{key}_{colordict[key][1]}', font=self.font)
					draw.rectangle([cx, cy,cx+text_width, cy+text_height],fill=(0,0,0))
					draw.text((cx, cy), f'{key}_{colordict[key][1]}', fill=tuple(colordict[key][0]), font=self.font)
				elif key==0:
					pass
				else:
					text_width, text_height = draw.textsize(f'{key}', font=self.font)
					draw.rectangle([cx, cy,cx+text_width, cy+text_height],fill=(0,0,0))
					draw.text((cx, cy), f'{key}', fill=tuple(colordict[key][0]), font=self.font)
		
		text_width, text_height = draw.textsize(f"{imagepath.strip('.bmp')[-4:]}", font=self.tickfont)
		draw.rectangle([0, 0,text_width, text_height],fill=(0,0,0))
		draw.text((0, 0), f"{imagepath.strip('.bmp')[-4:]}", fill=(255,255,255), font=self.tickfont)
		image=Image.open(imagepath)
		image_array = np.array(image)
		expanded_mask=np.array(maskimage)
		alpha=0.4
		foreground = (image_array * alpha + np.ones(image_array.shape) * (1 - alpha) * expanded_mask)
		image = foreground.astype('uint8')
		token=imagepath.strip(".bmp").split("/")[-1][-4:]
		cv2.imwrite(f"/mnt/ve_share/liushuai/panoptic-segment-anything/results/{token}.jpg",image)
		return image