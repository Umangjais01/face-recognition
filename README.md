************************************* 15-03-2024 ***********************************
 
    **** main.py ****
kajal sends base_64 string through "http://192.168.1.11.5001/login"
Umang is sending response as {"user_name":"user_name", "status":0/1}


    **** predictorfaces.py ****
    
changes
before :- it was asking from asktoopen from local system
now:- in predctor.py file take base_64 input as kajal's output i.e {data}


Error1:  like fc.weight/ fc.bias(...)
solution - check if model is present there or not


 


python3 -m venv myenv 
source myenv/bin/activate
pip install split-folders
pip install torch torchvision tqdm flask matplotlib requests opencv-python pandas

