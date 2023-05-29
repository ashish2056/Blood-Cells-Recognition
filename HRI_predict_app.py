import cv2
from flask import Flask, render_template, request,send_from_directory
import os
from darkflow.net.build import TFNet

options = {'model': 'cfg/tiny-yolo-voc-3c.cfg',
           'load': 3750,
           'threshold': 0.15,
           'gpu': 0.7}

tfnet = TFNet(options)

app = Flask(__name__)

app.config["IMAGE_UPLOADS"] = "D:\blood_counting\Automatic-Identification-and-Counting-of-Blood-Cells-master\images"
img = os.path.join('D:\blood_counting\Automatic-Identification-and-Counting-of-Blood-Cells-master\output',)

@app.route("/",)
def hello_world():
    return render_template('index.html')

@app.route("/", methods=["POST","GET"])
def predict(): 
    imagefile=request.files['imagefile']
    #full_filename = os.path.join(app.config['IMAGE_UPLOADS'], imagefile.filename)
    #image_file=imagefile.save(os.path.join(app.config["IMAGE_UPLOADS"], imagefile.filename))
   
    image_path= "./data/" + imagefile.filename
    imagefile.save(image_path) 

    rbc = 0
    wbc = 0
    platelets = 0
    

    C = []  # Center
    R = []  # Radius
    L = []  # Label

    im_name = 'HRI001'
    image = cv2.imread('data/' + imagefile.filename)

    for h in range(0, 2592, 890):
        for w in range(0, 3872, 1290):
            im = image[h:h + 890, w:w + 1290]
            output = tfnet.return_predict(im)

            RBC = 0
            WBC = 0
            Platelets = 0

            for prediction in output:
                label = prediction['label']
                confidence = prediction['confidence']

                tl = (prediction['topleft']['x'], prediction['topleft']['y'])
                br = (prediction['bottomright']['x'], prediction['bottomright']['y'])

                height, width, _ = image.shape
                center_x = int((tl[0] + br[0]) / 2)
                center_y = int((tl[1] + br[1]) / 2)
                center = (center_x + w, center_y + h)
                radius = int((br[0] - tl[0]) / 2)

                if label == 'RBC':
                    color = (255, 0, 0)
                    rbc = rbc + 1
                if label == 'WBC':
                    color = (0, 255, 0)
                    wbc = wbc + 1
                if label == 'Platelets':
                    color = (0, 0, 255)
                platelets = platelets + 1

                C.append(center)
                R.append(radius)
                L.append(label)

    record = []

    for i in range(0, len(C)):
        center = C[i]
        radius = R[i]
        label = L[i]

        if label == 'RBC':
            color = (255, 0, 0)
        elif label == 'WBC':
            color = (0, 255, 0)
        elif label == 'Platelets':
            color = (0, 0, 255)

        image = cv2.circle(image, center, radius, color, 5)
        font = cv2.FONT_HERSHEY_COMPLEX
        image = cv2.putText(image, label, (center[0] - 30, center[1] + 10), font, 1, color, 2)

    cv2.imwrite('output/' + imagefile.filename , image)
    new_width = 1000
    new_height = 600
    resized_image = cv2.resize(image, (new_width, new_height))
    cv2.imshow('Any',resized_image)
    if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

    return render_template('index.html',RBC=rbc ,WBC=wbc,Plat=platelets )        

if __name__ == '__main__':
    app.run(port=3000, debug=True)     
