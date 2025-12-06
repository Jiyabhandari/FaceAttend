############################################# IMPORTING ################################################
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
import tkinter.simpledialog as tsd
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import re

############################################# CONFIG ##################################################

# Recommended: install opencv-contrib-python so cv2.face is available:
# py -3.12 -m pip install opencv-contrib-python numpy pillow pandas

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Helper to build paths safely (uses forward slashes internally)
def p(path):
    return os.path.join(BASE_DIR, *path.split('/'))

############################################# FUNCTIONS ################################################

def assure_dir(path):
    """Ensure directory exists. Path may be a folder or file path."""
    if not path:
        return
    # If user provided a filepath, get directory component
    dir_path = path if path.endswith(os.sep) or '.' not in os.path.basename(path) else os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

##################################################################################

def tick():
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    clock.after(200, tick)

###################################################################################

def contact():
    mess._show(title='Contact us', message="Please contact us on : 'sum@gmail.com' ")

###################################################################################

def check_haarcascadefile():
    cascade_path = p("haarcascade_frontalface_default.xml")
    exists = os.path.isfile(cascade_path)
    if not exists:
        mess._show(title='Some file missing', message=f'Please ensure "{os.path.basename(cascade_path)}" is present in the project folder.')
        window.destroy()

###################################################################################

def save_pass():
    assure_dir(p("TrainingImageLabel/"))
    psd_path = p("TrainingImageLabel/psd.txt")
    if os.path.isfile(psd_path):
        with open(psd_path, "r", encoding="utf-8") as tf:
            key = tf.read()
    else:
        master.destroy()
        new_pas = tsd.askstring('Old Password not found', 'Please enter a new password below', show='*')
        if new_pas is None:
            mess._show(title='No Password Entered', message='Password not set!! Please try again')
        else:
            with open(psd_path, "w", encoding="utf-8") as tf:
                tf.write(new_pas)
            mess._show(title='Password Registered', message='New password was registered successfully!!')
            return

    op = old.get()
    newp = new.get()
    nnewp = nnew.get()

    if op == key:
        if newp == nnewp:
            with open(psd_path, "w", encoding="utf-8") as txf:
                txf.write(newp)
        else:
            mess._show(title='Error', message='Confirm new password again!!!')
            return
    else:
        mess._show(title='Wrong Password', message='Please enter correct old password.')
        return

    mess._show(title='Password Changed', message='Password changed successfully!!')
    master.destroy()

###################################################################################

def change_pass():
    global master
    master = tk.Tk()
    master.geometry("400x160")
    master.resizable(False, False)
    master.title("Change Password")
    master.configure(background="white")
    lbl4 = tk.Label(master, text='    Enter Old Password', bg='white', font=('comic', 12, ' bold '))
    lbl4.place(x=10, y=10)
    global old
    old = tk.Entry(master, width=25, fg="black", relief='solid', font=('comic', 12, ' bold '), show='*')
    old.place(x=180, y=10)
    lbl5 = tk.Label(master, text='   Enter New Password', bg='white', font=('comic', 12, ' bold '))
    lbl5.place(x=10, y=45)
    global new
    new = tk.Entry(master, width=25, fg="black", relief='solid', font=('comic', 12, ' bold '), show='*')
    new.place(x=180, y=45)
    lbl6 = tk.Label(master, text='Confirm New Password', bg='white', font=('comic', 12, ' bold '))
    lbl6.place(x=10, y=80)
    global nnew
    nnew = tk.Entry(master, width=25, fg="black", relief='solid', font=('comic', 12, ' bold '), show='*')
    nnew.place(x=180, y=80)
    cancel = tk.Button(master, text="Cancel", command=master.destroy, fg="black", bg="red", height=1, width=25,
                       activebackground="white", font=('comic', 10, ' bold '))
    cancel.place(x=200, y=120)
    save1 = tk.Button(master, text="Save", command=save_pass, fg="black", bg="#00fcca", height=1, width=25,
                      activebackground="white", font=('comic', 10, ' bold '))
    save1.place(x=10, y=120)
    master.mainloop()

#####################################################################################

def psw():
    assure_dir(p("TrainingImageLabel/"))
    psd_path = p("TrainingImageLabel/psd.txt")
    if os.path.isfile(psd_path):
        with open(psd_path, "r", encoding="utf-8") as tf:
            key = tf.read()
    else:
        new_pas = tsd.askstring('Old Password not found', 'Please enter a new password below', show='*')
        if new_pas is None:
            mess._show(title='No Password Entered', message='Password not set!! Please try again')
        else:
            with open(psd_path, "w", encoding="utf-8") as tf:
                tf.write(new_pas)
            mess._show(title='Password Registered', message='New password was registered successfully!!')
            return

    password = tsd.askstring('Password', 'Enter Password', show='*')
    if password == key:
        TrainImages()
    elif password is None:
        pass
    else:
        mess._show(title='Wrong Password', message='You have entered wrong password')

######################################################################################

def clear():
    txt.delete(0, 'end')
    res_text = "1)Take Images  >>>  2)Save Profile"
    message1.configure(text=res_text)

def clear2():
    txt2.delete(0, 'end')
    res_text = "1)Take Images  >>>  2)Save Profile"
    message1.configure(text=res_text)

#######################################################################################

def sanitize_filename(name: str):
    """Return a safe filename fragment (remove problematic characters)."""
    # keep letters, numbers, dash, underscore and spaces (convert multiple spaces to single)
    cleaned = re.sub(r'[^A-Za-z0-9 _-]', '', name).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.replace(' ', '_')

#######################################################################################

def TakeImages():
    check_haarcascadefile()
    columns = ['SERIAL NO.', '', 'ID', '', 'NAME']
    student_details_path = p("StudentDetails/StudentDetails.csv")
    training_image_dir = p("TrainingImage/")
    assure_dir(student_details_path)
    assure_dir(training_image_dir)

    # Compute serial number based on existing CSV rows
    serial = 0
    if os.path.isfile(student_details_path):
        with open(student_details_path, 'r', newline='', encoding='utf-8') as csvFile1:
            reader1 = csv.reader(csvFile1)
            rows = list(reader1)
            serial = len(rows) // 2  # keep old logic (this project stored alternating lines)
    else:
        # create header
        with open(student_details_path, 'a+', newline='', encoding='utf-8') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(columns)
        serial = 1

    Id = txt.get().strip()
    name = txt2.get().strip()

    # allow alphabetic and spaces only (improved)
    if not name or not all((c.isalpha() or c.isspace()) for c in name):
        res = "Enter Correct name (letters and spaces only)"
        message.configure(text=res)
        return

    if not Id:
        message.configure(text="Enter ID")
        return

    # sanitize for filenames
    safe_name = sanitize_filename(name)
    safe_id = sanitize_filename(Id)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        mess._show(title='Camera Error', message='Cannot open camera. Check camera or permissions.')
        return

    harcascadePath = p("haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(harcascadePath)
    sampleNum = 0
    max_samples = 100

    while True:
        ret, img = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sampleNum += 1
            fname = f"{safe_name}.{serial}.{safe_id}.{sampleNum}.jpg"
            out_path = os.path.join(training_image_dir, fname)
            cv2.imwrite(out_path, gray[y:y + h, x:x + w])
            cv2.imshow('Taking Images', img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        elif sampleNum > max_samples:
            break

    cam.release()
    cv2.destroyAllWindows()

    res = "Images Taken for ID : " + Id
    row = [serial, '', Id, '', name]
    with open(student_details_path, 'a+', newline='', encoding='utf-8') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    message1.configure(text=res)

########################################################################################

def TrainImages():
    check_haarcascadefile()
    assure_dir(p("TrainingImageLabel/"))

    # Create LBPH recognizer (requires opencv-contrib)
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except Exception as e:
        mess._show(title='OpenCV Error', message=f'Could not create LBPH recognizer. Ensure you installed opencv-contrib-python.\n\n{e}')
        return

    harcascadePath = p("haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(harcascadePath)

    faces, IDs = getImagesAndLabels(p("TrainingImage/"))
    if len(faces) == 0 or len(IDs) == 0:
        mess._show(title='No Registrations', message='Please Register someone first!!!')
        return

    try:
        recognizer.train(faces, np.array(IDs))
    except Exception as e:
        mess._show(title='Training Error', message=f'Error during training: {e}')
        return

    recognizer_path = p("TrainingImageLabel/Trainner.yml")
    recognizer.save(recognizer_path)
    res = "Profile Saved Successfully"
    message1.configure(text=res)
    try:
        message.configure(text='Total Registrations till now  : ' + str(IDs[0]))
    except:
        pass

############################################################################################3

def getImagesAndLabels(path):
    # path should be a directory
    imageDir = path
    imagePaths = []
    if not os.path.isdir(imageDir):
        return [], []

    for f in os.listdir(imageDir):
        fp = os.path.join(imageDir, f)
        if os.path.isfile(fp) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            imagePaths.append(fp)

    faces = []
    Ids = []
    for imagePath in imagePaths:
        try:
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            # Expect filename format: name.serial.id.sample.jpg -> ID is at index 2
            parts = os.path.basename(imagePath).split(".")
            if len(parts) >= 4:
                try:
                    ID = int(parts[2])
                except:
                    # fallback: attempt to parse numeric sequence if available
                    ID = 0
            else:
                ID = 0
            faces.append(imageNp)
            Ids.append(ID)
        except Exception as e:
            # skip unreadable files
            print(f"Skipping file {imagePath}: {e}")
    return faces, Ids

###########################################################################################

def TrackImages():
    check_haarcascadefile()
    assure_dir(p("Attendance/"))
    assure_dir(p("StudentDetails/"))

    # clear treeview
    for k in tv.get_children():
        tv.delete(k)

    recognizer = None
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except Exception as e:
        mess._show(title='OpenCV Error', message=f'Could not create LBPH recognizer. Ensure opencv-contrib-python is installed.\n\n{e}')
        return

    recognizer_path = p("TrainingImageLabel/Trainner.yml")
    if os.path.isfile(recognizer_path):
        recognizer.read(recognizer_path)
    else:
        mess._show(title='Data Missing', message='Please click on Save Profile to reset data!!')
        return

    harcascadePath = p("haarcascade_frontalface_default.xml")
    faceCascade = cv2.CascadeClassifier(harcascadePath)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        mess._show(title='Camera Error', message='Cannot open camera. Check camera or permissions.')
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', '', 'Name', '', 'Date', '', 'Time']
    student_details_path = p("StudentDetails/StudentDetails.csv")
    if os.path.isfile(student_details_path):
        try:
            df = pd.read_csv(student_details_path)
        except Exception as e:
            mess._show(title='Details Error', message=f'Could not read student details: {e}')
            cam.release()
            cv2.destroyAllWindows()
            return
    else:
        mess._show(title='Details Missing', message='Students details are missing, please check!')
        cam.release()
        cv2.destroyAllWindows()
        window.destroy()
        return

    attendance = None
    while True:
        ret, im = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            try:
                serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
            except Exception:
                serial, conf = -1, 1000
            if conf < 50:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = df.loc[df['SERIAL NO.'] == serial]['NAME'].values
                ID = df.loc[df['SERIAL NO.'] == serial]['ID'].values
                ID = str(ID)
                ID = ID[1:-1]
                bb = str(aa)
                bb = bb[2:-2]
                attendance = [str(ID), '', bb, '', str(date), '', str(timeStamp)]
            else:
                bb = 'Unknown'
                attendance = None
            cv2.putText(im, str(bb), (x, y + h), font, 1, (255, 255, 255), 2)

        cv2.imshow('Taking Attendance', im)
        if (cv2.waitKey(1) == ord('q')):
            break

    # write attendance if found
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    attendance_path = p(f"Attendance/Attendance_{date}.csv")
    if attendance:
        if os.path.isfile(attendance_path):
            with open(attendance_path, 'a+', newline='', encoding='utf-8') as csvFile1:
                writer = csv.writer(csvFile1)
                writer.writerow(attendance)
        else:
            with open(attendance_path, 'a+', newline='', encoding='utf-8') as csvFile1:
                writer = csv.writer(csvFile1)
                writer.writerow(col_names)
                writer.writerow(attendance)

    # populate treeview
    if os.path.isfile(attendance_path):
        with open(attendance_path, 'r', newline='', encoding='utf-8') as csvFile1:
            reader1 = csv.reader(csvFile1)
            i = 0
            for lines in reader1:
                i += 1
                if i > 1:
                    # insert alternate rows (keeps old behavior)
                    tv.insert('', 0, text=(str(lines[0]) + '   '), values=(str(lines[2]), str(lines[4]), str(lines[6])))

    cam.release()
    cv2.destroyAllWindows()

######################################## USED STUFFS ############################################

global key
key = ''

ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
day, month, year = date.split("-")

mont = {'01': 'January',
        '02': 'February',
        '03': 'March',
        '04': 'April',
        '05': 'May',
        '06': 'June',
        '07': 'July',
        '08': 'August',
        '09': 'September',
        '10': 'October',
        '11': 'November',
        '12': 'December'
        }

######################################## GUI FRONT-END ###########################################

window = tk.Tk()
window.geometry("1280x720")
window.resizable(True, False)
window.title("Attendance System")
window.configure(background='#f5f5f5')

frame1 = tk.Frame(window, bg="#B0B0B0")
frame1.place(relx=0.11, rely=0.17, relwidth=0.39, relheight=0.80)

frame2 = tk.Frame(window, bg="#B0B0B0")
frame2.place(relx=0.51, rely=0.17, relwidth=0.38, relheight=0.80)

message3 = tk.Label(window, text="Face Recognition Based Attendance Monitoring System", fg="#2b2d42", bg="#f5f5f5",
                    width=55, height=1, font=('comic', 29, ' bold '))
message3.place(x=10, y=10)

frame3 = tk.Frame(window, bg="#B0B0B0")
frame3.place(relx=0.55, rely=0.09, relwidth=0.15, relheight=0.07)

frame4 = tk.Frame(window, bg="#B0B0B0")
frame4.place(relx=0.33, rely=0.09, relwidth=0.25, relheight=0.07)

datef = tk.Label(frame4, text= day + "-" + mont[month] + "-" + year + "  |  ", fg="#4B4B60", bg="#f5f5f5", width=90,
                 height=1, font=('comic', 22, ' bold '))
datef.pack(fill='both', expand=2)

clock = tk.Label(frame3, fg="#4B4B60", bg="#f5f5f5", width=55, height=1, font=('comic', 22, ' bold '))
clock.pack(fill='both', expand=2)
tick()

head2 = tk.Label(frame2, text="                       For New Registrations                       ", fg="white", bg="#2b2d42",
                 font=('comic', 17, ' bold '))
head2.grid(row=0, column=0)

head1 = tk.Label(frame1, text="                       For Already Registered                       ", fg="white", bg="#2b2d42",
                 font=('comic', 17, ' bold '))
head1.place(x=0, y=0)

lbl = tk.Label(frame2, text="Enter ID", width=20, height=1, fg="black", bg="#B0B0B0", font=('comic', 17, ' bold '))
lbl.place(x=80, y=55)

txt = tk.Entry(frame2, width=32, fg="black", font=('comic', 15, ' bold '))
txt.place(x=30, y=88)

lbl2 = tk.Label(frame2, text="Enter Name", width=20, fg="black", bg="#B0B0B0", font=('comic', 17, ' bold '))
lbl2.place(x=80, y=140)

txt2 = tk.Entry(frame2, width=32, fg="black", font=('comic', 15, ' bold '))
txt2.place(x=30, y=173)

message1 = tk.Label(frame2, text="1)Take Images  >>>  2)Save Profile", bg="#B0B0B0", fg="black", width=39, height=1,
                    activebackground="#3ffc00", font=('comic', 15, ' bold '))
message1.place(x=7, y=230)

message = tk.Label(frame2, text="", bg="#B0B0B0", fg="black", width=39, height=1, activebackground="#3ffc00",
                   font=('comic', 16, ' bold '))
message.place(x=7, y=450)

lbl3 = tk.Label(frame1, text="Attendance", width=20, fg="black", bg="#B0B0B0", height=1, font=('comic', 17, ' bold '))
lbl3.place(x=100, y=115)

# Compute current total registrations
student_details_path = p("StudentDetails/StudentDetails.csv")
res = 0
if os.path.isfile(student_details_path):
    try:
        with open(student_details_path, 'r', newline='', encoding='utf-8') as csvFile1:
            reader1 = csv.reader(csvFile1)
            rows = list(reader1)
            res = (len(rows) // 2) - 1
            if res < 0:
                res = 0
    except Exception:
        res = 0
else:
    res = 0

message.configure(text='Total Registrations till now  : ' + str(res))

##################### MENUBAR #################################

menubar = tk.Menu(window, relief='ridge')
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label='Change Password', command=change_pass)
filemenu.add_command(label='Contact Us', command=contact)
filemenu.add_command(label='Exit', command=window.destroy)
menubar.add_cascade(label='Help', font=('comic', 29, ' bold '), menu=filemenu)

################## TREEVIEW ATTENDANCE TABLE ####################

tv = ttk.Treeview(frame1, height=13, columns=('name', 'date', 'time'))
tv.column('#0', width=82)
tv.column('name', width=130)
tv.column('date', width=133)
tv.column('time', width=133)
tv.grid(row=2, column=0, padx=(0, 0), pady=(150, 0), columnspan=4)
tv.heading('#0', text='ID')
tv.heading('name', text='NAME')
tv.heading('date', text='DATE')
tv.heading('time', text='TIME')

###################### SCROLLBAR ################################

scroll = ttk.Scrollbar(frame1, orient='vertical', command=tv.yview)
scroll.grid(row=2, column=4, padx=(0, 100), pady=(150, 0), sticky='ns')
tv.configure(yscrollcommand=scroll.set)

###################### BUTTONS ##################################

clearButton = tk.Button(frame2, text="Clear", command=clear, fg="#ffffff", bg="#3f72af", width=11,
                        activebackground="white", font=('comic', 11, ' bold '))
clearButton.place(x=335, y=86)
clearButton2 = tk.Button(frame2, text="Clear", command=clear2, fg="#ffffff", bg="#3f72af", width=11,
                         activebackground="white", font=('comic', 11, ' bold '))
clearButton2.place(x=335, y=172)
takeImg = tk.Button(frame2, text="Take Images", command=TakeImages, fg="white", bg="#3f72af", width=34, height=1,
                    activebackground="white", font=('comic', 15, ' bold '))
takeImg.place(x=30, y=300)
trainImg = tk.Button(frame2, text="Save Profile", command=psw, fg="white", bg="#28A745", width=34, height=1,
                     activebackground="white", font=('comic', 15, ' bold '))
trainImg.place(x=30, y=380)
trackImg = tk.Button(frame1, text="Take Attendance", command=TrackImages, fg="white", bg="#28A745", width=35, height=1,
                     activebackground="white", font=('comic', 15, ' bold '))
trackImg.place(x=30, y=50)
quitWindow = tk.Button(frame1, text="Quit", command=window.destroy, fg="#ffffff", bg="#f25c54", width=35, height=1,
                       activebackground="white", font=('comic', 15, ' bold '))
quitWindow.place(x=30, y=450)

##################### END ######################################

window.configure(menu=menubar)
window.mainloop()

####################################################################################################