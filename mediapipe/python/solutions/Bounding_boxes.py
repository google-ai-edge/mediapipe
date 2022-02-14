import cv2
import numpy as np

def run(frame, net, classes):
        
    height, width, _ = frame.shape   #height and width of the frame captured
        
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), swapRB = True, crop = False)
    net.setInput(blob)
    
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
    
    boxes = []       #stores the coordinates and measurements for the bounding box
    confidences = [] #Stores the confidence, i.e how much the object atches with a given class
    class_ids = []   #stores all the labels

    for output in layerOutputs:   #get ouput layers information
        for detection in output:  #extract information from each output (detection contains 85 parameters)
            
            scores = detection[5:] #prediction from all the classes, 6th element onwards
            
            class_id = np.argmax(scores) #extract location of the class with maximum confidence(index)
            confidence = scores[class_id] #extract the vaue of the confidence
            if confidence > 0.5:
                #these are normalised co-ordinates that is why we multiply them with heigth and width to
                #scale them back
                center_x = int(detection[0]*width) #the center x co-ordinate of the bounding box
                center_y = int(detection[1]*height) #the center y co-ordinate of the bounding box
                w = int(detection[2]*width)         #width of the bounding box
                h = int(detection[3]*height)        #height of the bounding box

                x = int(center_x - w/2)             #corner x co-ordinate
                y = int(center_y - h/2)             #corner y co-ordinate

                boxes.append([x, y, w, h])          #saves the co-ordinates and measurement in boxes[]
                confidences.append((float(confidence))) #saves the confidences of the classes
                class_ids.append(class_id)              #index of the classes detected
    
    #performs non-Max Supression on the classes with confidence greater then the threshold
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.2) 


    return indexes, boxes, class_ids, confidences

def boxing(frame, indexes, boxes, class_ids, confidences, classes, font):
    for i in indexes.flatten(): 
            x, y, w, h =  boxes[i] #co-ordinates if bounding boxes of final object after NMS
            label = str(classes[class_ids[i]]) #the name of the object detected
            confidence = str(round(confidences[i], 2)) #saves the confidence rounding it to 2 decimals
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) #bounda a rectangle around the object
            #shows the confidence and object name at top left
            cv2.putText(frame, label + " " + confidence, (x, y+20), font, 2, (255, 255, 255), 2)

    return frame