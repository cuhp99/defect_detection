import json
import numpy as np
import skimage.io
import cv2
import matplotlib.pyplot as plt
import PIL.Image
import PIL.ImageDraw

import os
import shutil

import cv2
import numpy as np

data_path = "dataset/Class1_label"
data_mask = "dataset/Class1_label_mask"
def make_map_label(data_path):
    # Loop through normal folder
            #mask_folder="Class1_label_mask"
            for file in os.listdir(data_path):
                if file.endswith("json"):
                    print(file)
                    # Read image file
                    file1 = os.path.join(data_path, file)

                    data=json.load(open(file1))

                    img_path = data['imagePath'].split('/')[-1]
                    img = skimage.io.imread(img_path)

                    def polygons_to_mask(img_shape, polygons):
                        '''
                        Máscara de generación de punto límite
                        :param img_shape: [h,w]
                             : polígonos param: formato de punto límite en JME de etiqueta [[x1, y1], [x2, y2], [x3, y3], ... [xn, yn]]
                        :return:
                        '''
                        mask = np.zeros(img_shape, dtype=np.uint8)
                        mask = PIL.Image.fromarray(mask)
                        xy = list(map(tuple, polygons))
                        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
                        mask = np.array(mask, dtype=bool)
                        return mask

                    def polygons_to_mask2(img_shape, polygons):
                        '''
                             Máscara de generación de punto límite
                        :param img_shape: [h,w]
                             : polígonos param: formato de punto límite en JME de etiqueta [[x1, y1], [x2, y2], [x3, y3], ... [xn, yn]]
                        :return:
                        '''
                        mask = np.zeros(img_shape, dtype=np.uint8)
                        polygons = np.asarray([polygons],np.int32)  # Esto debe ser int32, otros tipos que usan fillPoly informarán un error
                        # cv2.fillPoly (máscara, polígonos, 1) # Non-int32 informará un error
                        cv2.fillConvexPoly(mask, polygons, 1)  # Non-int32 informará un error
                        return mask

                    points = []
                    labels = []
                    for shapes in data['shapes']:
                        points.append(shapes['points'])
                        labels.append(shapes['label'])

                    mask0 = polygons_to_mask(img.shape[:2], points[0])
                    #mask1=polygons_to_mask(img.shape[:2],points[0])
                    mask_image = mask0.fromarray(mask_image)
                    mask_image.save(os.path.join(data_mask, file))


                    #plt.imshow(mask0.astype(np.uint8),'gray')

                    #plt.axis('off')
                    #plt.title('SegmentationObject:\n'+labels[0])
                    #plt.show()
                    #cv2.imwrite("mask0",mask0)
                    #plt.imsave("anh0.png",mask0,)
make_map_label(data_path)