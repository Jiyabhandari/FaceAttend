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

def load_attendance():
    """Load and display attendance records from today's CSV file in the treeview."""
    # Clear existing entries
    for k in tv.get_children():
        tv.delete(k)
    
    # Get today's date for attendance file
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    attendance_path = p(f"Attendance/Attendance_{date}.csv")
    
    # Load attendance records if file exists
    if os.path.isfile(attendance_path):
        try:
            with open(attendance_path, 'r', newline='', encoding='utf-8') as csvFile1:
                reader1 = csv.reader(csvFile1)
                next(reader1, None)  # Skip header
                for lines in reader1:
                    if len(lines) >= 7:
                        tv.insert('', 0, text=(str(lines[0]) + '   '), values=(str(lines[2]), str(lines[4]), str(lines[6])))
        except Exception as e:
            print(f"Error loading attendance: {e}")

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

    # Get today's date for attendance file
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    attendance_path = p(f"Attendance/Attendance_{date}.csv")
    
    # Track already marked IDs to avoid duplicates in same session
    marked_ids = set()
    # Track IDs that were just recorded in this session (for popup)
    just_recorded = set()
    
    # Load existing attendance to avoid duplicates
    if os.path.isfile(attendance_path):
        try:
            with open(attendance_path, 'r', newline='', encoding='utf-8') as csvFile1:
                reader1 = csv.reader(csvFile1)
                next(reader1, None)  # Skip header
                for row in reader1:
                    if len(row) > 0:
                        marked_ids.add(row[0])  # Store ID
        except Exception:
            pass
    
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
                try:
                    # Find student by serial number
                    student_row = df[df['SERIAL NO.'] == serial]
                    if not student_row.empty:
                        # Get first matching row
                        ID = str(student_row.iloc[0]['ID']).strip()
                        student_name = str(student_row.iloc[0]['NAME']).strip()
                        bb = student_name  # Display name
                        
                        # Write attendance immediately if not already marked
                        if ID not in marked_ids:
                            ts = time.time()
                            date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                            attendance = [str(ID), '', student_name, '', str(date), '', str(timeStamp)]
                            
                            # Write to file
                            try:
                                if os.path.isfile(attendance_path):
                                    with open(attendance_path, 'a+', newline='', encoding='utf-8') as csvFile1:
                                        writer = csv.writer(csvFile1)
                                        writer.writerow(attendance)
                                else:
                                    with open(attendance_path, 'a+', newline='', encoding='utf-8') as csvFile1:
                                        writer = csv.writer(csvFile1)
                                        writer.writerow(col_names)
                                        writer.writerow(attendance)
                                
                                # Mark as recorded
                                marked_ids.add(ID)
                                
                                # Update treeview immediately (force update)
                                tv.insert('', 0, text=(str(ID) + '   '), values=(student_name, str(date), str(timeStamp)))
                                window.update()  # Force UI update
                                
                                # Show popup notification only once per ID
                                if ID not in just_recorded:
                                    mess._show(title='Attendance Recorded ✓', message=f'Attendance successfully recorded!\n\nName: {student_name}\nID: {ID}\nTime: {timeStamp}\nDate: {date}')
                                    just_recorded.add(ID)
                                
                                # Show success message on camera feed
                                bb = student_name + ' - ✓ RECORDED!'
                                
                                # Show confirmation in console
                                print(f"✓ Attendance recorded: {student_name} (ID: {ID}) at {timeStamp}")
                            except Exception as write_error:
                                bb = bb + ' - ✗ ERROR!'
                                print(f"Error writing attendance: {write_error}")
                                mess._show(title='Error', message=f'Failed to save attendance:\n{write_error}')
                        else:
                            # Already marked today
                            bb = bb + ' (Already Marked)'
                    else:
                        bb = 'Unknown (Serial: ' + str(serial) + ')'
                except Exception as e:
                    bb = 'Error: ' + str(e)
                    print(f"Error in recognition: {e}")
            else:
                bb = 'Unknown (Conf: ' + str(int(conf)) + ')'
            
            # Display name and status on camera feed
            # Use green color for successful recording, white for others
            if 'RECORDED' in bb or '✓' in bb:
                color = (0, 255, 0)  # Green for success
                thickness = 2
            elif 'Already Marked' in bb:
                color = (0, 165, 255)  # Orange for already marked
                thickness = 2
            else:
                color = (255, 255, 255)  # White for others
                thickness = 1
            
            cv2.putText(im, str(bb), (x, y + h), font, 1, color, thickness)
            
            # Show status at top of frame for better visibility
            if 'RECORDED' in bb or '✓' in bb:
                cv2.putText(im, 'ATTENDANCE SAVED TO FILE!', (10, 30), font, 1, (0, 255, 0), 2)
                cv2.putText(im, 'Check the table on the left', (10, 60), font, 0.7, (0, 255, 0), 2)

        cv2.imshow('Taking Attendance', im)
        if (cv2.waitKey(1) == ord('q')):
            break

    # Refresh treeview with all attendance records for today
    if os.path.isfile(attendance_path):
        # Clear existing entries
        for k in tv.get_children():
            tv.delete(k)
        # Reload all entries
        with open(attendance_path, 'r', newline='', encoding='utf-8') as csvFile1:
            reader1 = csv.reader(csvFile1)
            next(reader1, None)  # Skip header
            for lines in reader1:
                if len(lines) >= 7:
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
window.configure(background='#2d420a')

frame1 = tk.Frame(window, bg="#c79cff")
frame1.place(relx=0.11, rely=0.17, relwidth=0.39, relheight=0.80)

frame2 = tk.Frame(window, bg="#c79cff")
frame2.place(relx=0.51, rely=0.17, relwidth=0.38, relheight=0.80)

message3 = tk.Label(window, text="Face Recognition Based Attendance Monitoring System", fg="white", bg="#2d420a",
                    width=55, height=1, font=('comic', 29, ' bold '))
message3.place(x=10, y=10)

frame3 = tk.Frame(window, bg="#c4c6ce")
frame3.place(relx=0.52, rely=0.09, relwidth=0.09, relheight=0.07)

frame4 = tk.Frame(window, bg="#c4c6ce")
frame4.place(relx=0.36, rely=0.09, relwidth=0.16, relheight=0.07)

datef = tk.Label(frame4, text=day + "-" + mont[month] + "-" + year + "  |  ", fg="#ff61e5", bg="#2d420a", width=55,
                 height=1, font=('comic', 22, ' bold '))
datef.pack(fill='both', expand=1)

clock = tk.Label(frame3, fg="#ff61e5", bg="#2d420a", width=55, height=1, font=('comic', 22, ' bold '))
clock.pack(fill='both', expand=1)
tick()

head2 = tk.Label(frame2, text="                       For New Registrations                       ", fg="black", bg="#00fcca",
                 font=('comic', 17, ' bold '))
head2.grid(row=0, column=0)

head1 = tk.Label(frame1, text="                       For Already Registered                       ", fg="black", bg="#00fcca",
                 font=('comic', 17, ' bold '))
head1.place(x=0, y=0)

lbl = tk.Label(frame2, text="Enter ID", width=20, height=1, fg="black", bg="#c79cff", font=('comic', 17, ' bold '))
lbl.place(x=80, y=55)

txt = tk.Entry(frame2, width=32, fg="black", font=('comic', 15, ' bold '))
txt.place(x=30, y=88)

lbl2 = tk.Label(frame2, text="Enter Name", width=20, fg="black", bg="#c79cff", font=('comic', 17, ' bold '))
lbl2.place(x=80, y=140)

txt2 = tk.Entry(frame2, width=32, fg="black", font=('comic', 15, ' bold '))
txt2.place(x=30, y=173)

message1 = tk.Label(frame2, text="1)Take Images  >>>  2)Save Profile", bg="#c79cff", fg="black", width=39, height=1,
                    activebackground="#3ffc00", font=('comic', 15, ' bold '))
message1.place(x=7, y=230)

message = tk.Label(frame2, text="", bg="#c79cff", fg="black", width=39, height=1, activebackground="#3ffc00",
                   font=('comic', 16, ' bold '))
message.place(x=7, y=450)

lbl3 = tk.Label(frame1, text="Attendance", width=20, fg="black", bg="#c79cff", height=1, font=('comic', 17, ' bold '))
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

clearButton = tk.Button(frame2, text="Clear", command=clear, fg="black", bg="#ff7221", width=11,
                        activebackground="white", font=('comic', 11, ' bold '))
clearButton.place(x=335, y=86)
clearButton2 = tk.Button(frame2, text="Clear", command=clear2, fg="black", bg="#ff7221", width=11,
                         activebackground="white", font=('comic', 11, ' bold '))
clearButton2.place(x=335, y=172)
takeImg = tk.Button(frame2, text="Take Images", command=TakeImages, fg="white", bg="#6d00fc", width=34, height=1,
                    activebackground="white", font=('comic', 15, ' bold '))
takeImg.place(x=30, y=300)
trainImg = tk.Button(frame2, text="Save Profile", command=psw, fg="white", bg="#6d00fc", width=34, height=1,
                     activebackground="white", font=('comic', 15, ' bold '))
trainImg.place(x=30, y=380)
trackImg = tk.Button(frame1, text="Take Attendance", command=TrackImages, fg="black", bg="#3ffc00", width=35, height=1,
                     activebackground="white", font=('comic', 15, ' bold '))
trackImg.place(x=30, y=50)
quitWindow = tk.Button(frame1, text="Quit", command=window.destroy, fg="black", bg="#eb4600", width=35, height=1,
                       activebackground="white", font=('comic', 15, ' bold '))
quitWindow.place(x=30, y=450)

##################### END ######################################

window.configure(menu=menubar)

# Load existing attendance records on startup
load_attendance()

window.mainloop()

####################################################################################################
