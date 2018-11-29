import glob
import xml.etree.ElementTree as ET

#for train
txt_file = open('pedestrian.txt','w')

xmls=glob.glob('xmls/*.xml')
for xml in xmls:
	print(xml)
	name_file=xml.split('/')[-1][:-4]
	txt_file.write(f'/data/ruben/git/xxcv/data/pedestrian/images/{name_file}.jpg'+' ')
	tree = ET.parse(xml)
	objects = []
	for obj in tree.findall('object'):
		obj_struct = {}
		obj_struct['name'] = obj.find('name').text
		#obj_struct['pose'] = obj.find('pose').text
		#obj_struct['truncated'] = int(obj.find('truncated').text)
		#obj_struct['difficult'] = int(obj.find('difficult').text)
		bbox = obj.find('bndbox')
		obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
							  int(float(bbox.find('ymin').text)),
							  int(float(bbox.find('xmax').text)),
							  int(float(bbox.find('ymax').text))]
		objects.append(obj_struct)
	num_obj = len(objects)
	txt_file.write(str(num_obj)+' ')
	for result in objects:
		class_name = result['name']
		bbox = result['bbox']
		class_name = 1
		txt_file.write(str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])+' '+str(class_name)+' ')
	txt_file.write('\n')
	#if count == 10:
	#    break
txt_file.close()