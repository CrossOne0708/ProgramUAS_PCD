from datetime import datetime
import cv2, os, numpy as np
from tkinter import *
import tkinter as tk
from tkinter import ttk
import time
from PIL import ImageTk, Image

def selesai1():
    intructions.config(text="REKAM DATA TELAH SELESAI!", font="Roboto", fg="white", bg="green")
def selesai2():
    intructions.config(text="TRAINING WAJAH TELAH SELESAI!", font="Roboto", fg="white", bg="blue")
def selesai3():
    intructions.config(text="TERIMAKASIH ABSENSI TELAH DILAKUKAN",font="Roboto", fg="white", bg="orange")

def apply_texture_filter(image):
    # Konversi gambar ke mode grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Menggunakan filter tekstur (Contoh: Median Blur)
    filtered = cv2.medianBlur(gray, 5)

    # Konversi kembali ke mode BGR
    filtered_frame = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

    return filtered_frame

def rekamDataWajah():
    wajahDir = 'datawajah'
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eyeDetector = cv2.CascadeClassifier('haarcascade_eye.xml')
    faceID = entry2.get()
    nama = entry1.get()
    nim = entry2.get()
    kelas = entry3.get()
    Mata_Kuliah = entry4.get()
    ambilData_ori = 1
    ambilData_HE = 61
    ambilData_texture = 121
    
    # Membuat subdirektori jika belum ada
    sub_dir = os.path.join(wajahDir, nama.lower().replace(" ", "-"))
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    
    while True:
        retV, frame = cam.read()
        abuabu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetector.detectMultiScale(abuabu, 1.3, 5)
        
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            namaFile1 = str(nim) + '_' + str(nama) + '_' + str(kelas) + '_' + str(ambilData_ori) +'.jpg'
            file_path1 = os.path.join(sub_dir, namaFile1)
            cv2.imwrite(file_path1, frame)

            # Histogram equalization
            namaFile2 = str(nim) + '_' + str(nama) + '_' + str(kelas) + '_' + str(ambilData_HE) +'.jpg'
            file_path2 = os.path.join(sub_dir, namaFile2)
            equalized = cv2.equalizeHist(abuabu)
            equalized_frame = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(file_path2, equalized_frame)

            # Filter berdasarkan tekstur
            namaFile3 = str(nim) + '_' + str(nama) + '_' + str(kelas) + '_' + str(ambilData_texture) +'.jpg'
            file_path3 = os.path.join(sub_dir, namaFile3)
            filtered_frame = apply_texture_filter(frame)
            cv2.imwrite(file_path3, filtered_frame)

            ambilData_ori += 1
            ambilData_HE += 1
            ambilData_texture += 1

            roiabuabu = equalized[y:y + h, x:x + w]
            roiwarna = equalized_frame[y:y + h, x:x + w]
            eyes = eyeDetector.detectMultiScale(roiabuabu)
            for (xe, ye, we, he) in eyes:
                cv2.rectangle(roiwarna, (xe, ye), (xe + we, ye + he), (0, 255, 255), 1)
        
        cv2.imshow('TEKAN Q UNTUK KELUAR', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # jika menekan tombol q akan berhenti
            break
        elif ambilData_ori > 60:
            break
    
    selesai1()
    cam.release()
    cv2.destroyAllWindows()

IMAGE_SIZE = (200,200)
BATCH_SIZE = 32
SEED = 999

def trainface():
    import os
    base_dir = 'datawajah'
    print(os.listdir(base_dir))

    # Menggunakan ImageDataGenerator untuk preprocessing
    import tensorflow as tf

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        validation_split=0.2
    )

    # Menyiapkan data train dan data validation
    train_data = datagen.flow_from_directory(
        base_dir,
        class_mode='categorical',
        subset='training',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    valid_data = datagen.flow_from_directory(
        base_dir,
        class_mode='categorical',
        subset='validation',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    # Image Augmentation
    data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal",
                        input_shape=(IMAGE_SIZE[0],
                                    IMAGE_SIZE[1],
                                    3)),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.Rescaling(1./255)
    ]
    )

    # Membuat arsitektur model CNN
    cnn_model = tf.keras.models.Sequential([
    data_augmentation,
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
    ])

    # Compiling model
    cnn_model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    # Training model CNN
    cnn_hist = cnn_model.fit(
        train_data,
        epochs=20,
        validation_data = valid_data
    )

    # Simpan model ke dalam file model.h5
    cnn_model.save('model.h5')


def trainingWajah():
    wajahDir = 'datawajah'
    latihDir = 'latihwajah'

    def getImageLabel(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        faceIDs = []
        for imagePath in imagePaths:
            PILimg = Image.open(imagePath).convert('L')
            imgNum = np.array(PILimg, 'uint8')
            faceID = int(os.path.split(imagePath)[-1].split('_')[0])
            faces = faceDetector.detectMultiScale(imgNum)
            for (x, y, w, h) in faces:
                faceSamples.append(imgNum[y:y + h, x:x + w])
                faceIDs.append(faceID)
            return faceSamples, faceIDs

    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces, IDs = getImageLabel(wajahDir)
    faceRecognizer.train(faces, np.array(IDs))
    # simpan
    faceRecognizer.write(latihDir + '/training.xml')
    selesai2()

def markAttendance(name):
    with open("PRESENSI.csv", 'r+') as f:
        namesDatalist = f.readlines()
        namelist = []
        yournim = entry2.get()
        yourclass = entry3.get()
        yourMata_Kuliah = entry4.get()
        for line in namesDatalist:
            entry = line.split(';')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{yourclass},{yournim},{dtString}, {yourMata_Kuliah}')

def absensiWajah():
    import tensorflow as tf
    wajahDir = 'datawajah'
    latihDir = 'latihwajah'
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faceRecognizer = tf.keras.models.load_model('model.h5')
    font = cv2.FONT_HERSHEY_SIMPLEX

    yourname = entry1.get()
    names = []
    names.append(yourname)
    minWidth = 0.1 * cam.get(3)
    minHeight = 0.1 * cam.get(4)

    while True:
        retV, frame = cam.read()
        frame = cv2.flip(frame, 1)
        abuabu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetector.detectMultiScale(abuabu, 1.2, 5, minSize=(round(minWidth), round(minHeight)), )
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_img = abuabu[y:y+h, x:x+w]
            resized_img = cv2.resize(face_img, IMAGE_SIZE)
            normalized_img = resized_img / 255.0
            input_img = np.expand_dims(normalized_img, axis=0)
            prediction = faceRecognizer.predict(input_img)[0]
            id = np.argmax(prediction)
            confidence = prediction[id]

            if confidence < 0.5:
                id = names[0]
                confidence = "  {0}%".format(round(100 - confidence * 100))
            else:
                id = "Tidak Diketahui"
                confidence = "  {0}%".format(round(100 - confidence * 100))

            cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(confidence), (x + 5, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow('TEKAN Q UNTUK KELUAR', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    markAttendance(id)
    selesai3()
    cam.release()
    cv2.destroyAllWindows()

# GUI
root = tk.Tk()

root.attributes("-fullscreen", True)
# mengatur canvas (window tkinter)
root.title("Smart Attendance System")
root.wm_iconbitmap('Logo.ico') # untuk tampilkan icon di window border

canvas = tk.Canvas(root, width=1600, height=800)
canvas.grid(columnspan=2, rowspan=8)
canvas.configure(bg="#2ECC71")


##################################################################################

def tick():
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    clock.after(200,tick)

######################################## USED STUFFS ############################################

global key
key = ''

ts = time.time()
date = datetime.fromtimestamp(ts).strftime('%d %m %Y')
day, month, year = date.split(" ")

mont = {'01': 'Januari',
        '02': 'Februari',
        '03': 'Maret',
        '04': 'April',
        '05': 'Mei',
        '06': 'Juni',
        '07': 'Juli',
        '08': 'Augustus',
        '09': 'September',
        '10': 'Oktober',
        '11': 'November',
        '12': 'Desember'
        }
######################################################################################################
# judul
judul = tk.Button(root, text="SISTEM ABSENSI OTOMATIS", font=("Berlin Sans FB", 34), bg="#03A678", fg="white")
canvas.create_window(700, 100, window=judul)
judul.place(relx=0.005, rely=0.01, relwidth=0.99, relheight=0.1)
######################################################################################################
gambar3 = PhotoImage(file="logo baronas.png")
gambarlabel3 = Label(root, image=gambar3)
gambarlabel3.place(relx=0.03, rely=0.14, relwidth=0.05, relheight=0.10)
######################################################################################################
gambar4 = PhotoImage(file="merapi team.png")
gambarlabel4 = Label(root, image=gambar4)
gambarlabel4.place(relx=0.099, rely=0.14, relwidth=0.05, relheight=0.11)
######################################################################################################
#credit
made = tk.Label(root, text="Created By: MERAPI TEAM BARONAS ITS 2022", font=("Calibri", 13, "italic"), bg="#2ECC71", fg="black")
canvas.create_window(700, 30, window=made)
made.place(relx=0.7, rely=0.9, relwidth=0.3, relheight=0.1)

################## Kolom Nama Mahasiswa #################################################################
# for entry data nama
entry1 = tk.Entry (root, font="Roboto")
canvas.create_window(457, 170, height=25, width=411, window=entry1)
entry1.place(relx=0.185, rely=0.35, relwidth=0.3, relheight=0.035)

label1 = tk.Label(root, text="Nama Mahasiswa", font="Roboto", fg="white", bg="#2ECC71")
canvas.create_window(110,170, window=label1)
label1.place(relx=0.045, rely=0.35, relwidth=0.1, relheight=0.05)

################### Kolom NIM ############################################################################
# for entry data nim
entry2 = tk.Entry (root, font="Roboto")
canvas.create_window(457, 210, height=25, width=411, window=entry2)
entry2.place(relx=0.185, rely=0.4, relwidth=0.3, relheight=0.035)

label2 = tk.Label(root, text="NIM", font="Roboto", fg="white", bg="#2ECC71")
canvas.create_window(60, 210, window=label2)
label2.place(relx=0.01, rely=0.4, relwidth=0.1, relheight=0.05)

##################### Kelas ################################################################################
# for entry data kelas
entry3 = tk.Entry (root, font="Roboto")
canvas.create_window(457, 250, height=25, width=411, window=entry3)
entry3.place(relx=0.185, rely=0.45, relwidth=0.3, relheight=0.035)

label3 = tk.Label(root, text="Kelas", font="Roboto", fg="white", bg="#2ECC71")
canvas.create_window(65, 250, window=label3)
label3.place(relx=0.015, rely=0.45, relwidth=0.1, relheight=0.05)


########################################## Combo Box pemilihan mata kuliah ##################################


# label
ttk.Label(root, text="Mata Kuliah", background="#2ECC71", foreground="white",
          font=("Roboto", 12)).place(relx=0.05, rely=0.5, relwidth=0.1, relheight=0.05)

# Combobox creation
entry4 = tk.StringVar()
monthchoosen = ttk.Combobox(root, width=25, textvariable=entry4)

# Adding combobox drop down list
monthchoosen['values'] = ('Sistem Tertanam dan Internet of Things IoT',
                          'Sistem Kecerdasan Buatan',
                          'Sistem Komunikasi',
                          'Praktikum Sistem Komunikasi',
                          'Pengolahan Citra Digital',
                          'Praktikum Pengolahan Citra Digital',
                          'Sistem Operasi',
                          'Etika Profesi dan Rekayasa',
                          'PPKN')

monthchoosen.place(relx=0.185, rely=0.5, relwidth=0.3, relheight=0.035)
monthchoosen.current()
#############################################################################################################
global intructions



# tombol untuk rekam data wajah
# for entry data kelas
label5 = tk.Label(root, text="Status Kehadiran", font="Roboto", fg="white", bg="#2ECC71")
canvas.create_window(65, 250, window=label5)
label5.place(relx=0.042, rely=0.3, relwidth=0.1, relheight=0.05)
intructions = tk.Label(root, text="SELAMAT DATANG", font=("Calibri",15, 'bold'),fg="white",bg="#03A678")
canvas.create_window(370, 300, window=intructions)
intructions.place(relx=0.185, rely=0.3, relwidth=0.3, relheight=0.035)

Rekam_text = tk.StringVar()
Rekam_btn = tk.Button(root, textvariable=Rekam_text,
                      bd=10,
                      font=("Roboto",12,'bold'),
                      bg="#65c6bb",
                      fg="white",
                      height=1,
                      width=15,
                      command=rekamDataWajah)
Rekam_btn.place(relx=0.05, rely=0.8, relwidth=0.2, relheight=0.05)
Rekam_text.set("AMBIL GAMBAR")

# tombol untuk training wajah
Rekam_text1 = tk.StringVar()
Rekam_btn1 = tk.Button(root, textvariable=Rekam_text1,
                       bd=10,
                       font=("Roboto",12,'bold'),
                       bg="#65c6bb",
                       fg="white",
                       height=1,
                       width=15,
                       
                       command=trainface
                       )
Rekam_btn1.place(relx=0.3, rely=0.8, relwidth=0.2, relheight=0.05)
Rekam_text1.set("LATIH WAJAH")

# tombol absensi dengan wajah
Rekam_text2 = tk.StringVar()
Rekam_btn2 = tk.Button(root, textvariable=Rekam_text2,
                       bd=10,
                       font=("Roboto",12,'bold'),
                       bg="#65c6bb",
                       fg="white",
                       height=1,
                       width=20,
                       command=absensiWajah)
Rekam_btn2.place(relx=0.55, rely=0.8, relwidth=0.2, relheight=0.05)
Rekam_text2.set("ABSEN OTOMATIS")
##############################################################################################
# Tombol Exit
r=Button(root, text="EXIT",
         bd=10,
         command=quit,
         font=('times new roman',12),
         bg="red",
         fg="white",
         height=1,
         width= 10)
r.place(relx=0.83, rely=0.8, relwidth=0.1, relheight=0.06)
###############################################################################################
frame3 = tk.Frame(root, bg="#65c6bb")
frame3.place(relx=0.897, rely=0.17, relwidth=0.1, relheight=0.07)

frame4 = tk.Frame(root, bg="#65c6bb")
frame4.place(relx=0.81, rely=0.12, relwidth=0.2, relheight=0.07)

clock = tk.Label(frame3, fg="white", bg="#2ECC71", width=57, height=2, font=('calibri', 22, ' bold '))
clock.pack(fill='both', expand=1)
tick()

datef = tk.Label(frame4, text = (day+" "+mont[month]+" "+year), fg="white", bg="#2ECC71", width=70,
                 height=2,font=('calibri', 22, ' bold '))
datef.pack(fill='both', expand=1)

root.mainloop()
