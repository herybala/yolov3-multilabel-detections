import argparse
from sys import platform

from models import *
from utils import torch_utils
from utils.datasets import *
from utils.utils import *
import numpy as np

def detect_with_double_options(save_txt=True, save_img=True):
	img_size = opt.img_size
	out, source, weights = opt.output, opt.source, opt.weights
	# Initialize
	device = torch_utils.select_device(device = opt.device)
	if os.path.exists(out):
		shutil.rmtree(out)
	os.makedirs(out)
	# Initialize model
	model = Darknet(opt.cfg, img_size)
	# Load weights
	if weights.endswith('.pt'): # pytorch format
		model.load_state_dict(torch.load(weights, map_location=device)['model'])
	else: # Darknet format
		_ = load_darknet_weights(model, weights)
	# Eval mode
	model.to(device).eval()
	# Set dataloader
	if save_img:
		dataset = LoadImages(source, img_size=img_size)
	# Get classes and colors
	classes = load_classes(parse_data_cfg(opt.data)['names'])
	colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
	# Run inference
	results = dict()
	t0 = time.time()
	iou_thres = 0.5
	for path, img, im0s, vid_cap in dataset:
		t = time.time()
		# Get detections
		img = torch.from_numpy(img).to(device)
		if img.ndimension() == 3:
			img = img.unsqueeze(0)
		pred = model(img)[0]
		pred = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)
		# Process detections
		for i, det in enumerate(pred):
			p, s, im0 = path, '', im0s
			save_path = str(Path(out) / Path(p).name)
			s += '%gx%g' % img.shape[2:]  # print string
			if det is not None and len(det):
				# Rescale boxes from img_size to im0 size
				det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
				double_det_list = list()
				# Print results
				for c in det[:, -1].unique():
					n = (det[:, -1] == c).sum()  # detections per class
					s += '%g %ss, ' % (n, classes[int(c)])  # add to string
				if det.shape[0] > 1:
					detection = det
					for l in range(det.shape[0]):
						d = 0
						if detection.shape[0] > l+1:
							for k in range(int(l+1), detection.shape[0]):
								box1 = detection[l][0:4]
								box2 = detection[k-d][0:4]
								scorel = detection[l][4]
								scorek = detection[k-d][4]
								iou = get_iou(box1=box1, box2=box2)
								if iou == 1.0:
									pass
								elif iou >= iou_thres: # Double detections in one object
									if scorel > scorek:# remove det[k]
										cl_parent = detection[k-d][-1]
										conf_parent = detection[k-d][4]
										parent = torch.cat([conf_parent.unsqueeze(0),cl_parent.unsqueeze(0)])
										# Write results for detection[l]
										double_det_list.append(list(detection[l]))
										for *xyxy, conf, _, cls in detection[l].unsqueeze(0):
											label = '%s %.2f %c% s %.2f' % (classes[int(cls)], conf,"\n", classes[int(cl_parent)], conf_parent)
											plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
										detection = torch.cat([detection[:k-d],detection[k-d + 1:]])
										d += 1
									else:# remove det[l]
										cl_parent = detection[l][-1]
										conf_parent = detection[l][4]
										parent = torch.cat([conf_parent.unsqueeze(0),cl_parent.unsqueeze(0)])
										double_det_list.append(list(detection[k-d]))
										for *xyxy, conf, _, cls in detection[k-d].unsqueeze(0):
											label = '%s %.2f %c% %s %.2f' % (classes[int(cls)], conf,'\n', classes[int(cl_parent)], conf_parent)
											plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
										detection = torch.cat([detection[:l],detection[l + 1:]])
										d += 1
						else:
							continue
					# Plot other detections in detection variable
					for d in detection:
						xyxy = [d[0], d[1], d[2], d[3]]
						cls = d[-1]
						conf = d[4]
						if not double_det_list:
							label = '%s %.2f' % (classes[int(cls)], conf)
							plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
						else:
							if list(d) in double_det_list:
								pass
							else:
								label = '%s %.2f' % (classes[int(cls)], conf)
								plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
				else:
					for *xyxy, conf, _, cls in det:
						label = '%s %.2f' % (classes[int(cls)], conf)
						plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
				if save_img:
					cv2.imwrite(save_path, im0)
				if save_txt:
					for *xyxy, conf, _, cls in det:
						with open(save_path.split('.jpg')[0] + '.txt', 'a') as file:
							file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

			print('%sDone. (%.3fs)' % (s, time.time() - t), '\n')
	print('Done. (%.3fs)' % (time.time() - t0))

def main():
	pass
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect_with_double_options()