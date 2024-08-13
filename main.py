from flask import Flask, request, jsonify
import os
import Predictorfaces
import Trainerfaces
import cenrted_final
import base64
from PIL import Image
from io import BytesIO
import shutil
# from Trainerfaces import prepare_and_train_model 
app = Flask(__name__)

@app.route('/', methods=['GET'])
def testing():
    print("JAIS TEST FUNCTION")
    return " Server is Running"
# Route for handling GET requests
@app.route('/signup', methods=['POST'])
def signup():
    print('start ID:')

    data = request.get_json()
    images = data['base64_images']
    employee_id = data['employee_id']
    employee_name = data['employee_name']

    # print('Image:', images)
    print('Received name:',employee_name)
    print('Received ID:',employee_id)
    print("Length of array:", len(images))
    # return "jais"
    print("JAIS1 Creatting paths for dataset FUNCTION")

    # pwd_path = os.getcwd()
    pwd_path = os.getcwd()
    print("folder_path 1: ", pwd_path)

    dataset_directory = "dataset"
    employee_directory = os.path.join(dataset_directory, employee_id)
    if not os.path.exists(employee_directory):
        os.makedirs(employee_directory)

    # Change to the dataset directory
    os.chdir(dataset_directory)

    
    in_dataset_folder = os.getcwd()
    print("Current directory after changing to 'dataset': ", in_dataset_folder)

    # Ensure the employee directory exists
    
    if os.path.exists(employee_id):
        # Delete the folder if it exists
        shutil.rmtree(employee_id)
    
    os.makedirs(employee_id)

    # Change to the employee directory
    os.chdir(employee_id)
    employee_folder = os.getcwd()
    print("Current directory after changing to employee folder: ", employee_folder)
   
    # #    Uncomment to make the aligned folder
    # output_folder = f"/home/umang/Desktop/working/faces/aligned/{employee_id}"
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)


    # Decode and save each image
    for i, image in enumerate(images):
        image_data = base64.b64decode(image)
        image_raw = Image.open(BytesIO(image_data))
        image_filename = f"{employee_name}_{i+1}.jpg"
        image_raw.save(image_filename)
        print(f"Saved image: {image_filename}")

        # result = cenrted_final.Face_Alignment(image_filename, output_folder)
        # print (f"status : {result}")
        # if result == 0:
        #     print("Face not found , stopping further processing.")
        #     break

    if len(images) == 32:
        result = 1
    else:
        result = 0
    print("RESULT:", result)
    
    # reseting os directory
    os.chdir(pwd_path)

    Trainerfaces.prepare_and_train_model() 
    print("doneeeeeeeeeeeeeeeeeeeeeeeeee")     
    response= {
        'status': result,
        'emp_id': employee_id,
        'emp_name': employee_name
    }
    return jsonify(response)
# Route for handling GET requests
@app.route('/validation', methods=['POST'])
def login():
    # Your logic to fetch or generate data for GET requests
    print('start ID Login:')
    data = request.get_json()
    # print('Received data:', data)
    # images = data.get('images', [])  # Use get() to avoid KeyError if 'images' is missing
    # id = data.get('id', '')
    # name = data.get('name', '')
    images = data['base64_image']
    id = data['employee_id']
    name = data['employee_name']
    print("Length of array:", len(images))
    # print('Image:', images)
    print('Received ID:',id)
    # print('Received name:',name)
    # return "jais"

    response_data_list = []  # Store response data for each image
    for image_base64 in images:
        # image_base64 = data['images']
        response_data = Predictorfaces.predict_from_base64(image_base64)  
        response_data_list.append(response_data)
        # print("KK")
        print("percentage ",response_data)

    if response_data == "Unknown":
        employee_status = 0
    else: 
        employee_status = 1
#     employee_status = any(response_data != "Unknown" for response_data in response_data_list)

    response= {
        'status': employee_status,
        'emp_id': id,
        'emp_name':name
    }
    print("DONEEE")
    return jsonify(response)
    # return jsonify(employee_name=name, employee_status=int(employee_status), employee_phone="+91123456789")
    
    # return jsonify(employee_name=response_data, employee_status=employee_status, employee_phone="+91123456789")


if __name__ == '__main__': 
   
    # testing()
    print("Reyana")
    app.run(host='0.0.0.0', port=5001, debug=True) 
    print("JAIS")
    

