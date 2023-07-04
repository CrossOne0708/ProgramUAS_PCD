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
    ambilData_HE = 151
    ambilData_texture = 301
    ambilData_flipped = 451
    
    # Membuat subdirektori jika belum ada
    sub_dir = os.path.join(wajahDir, nama.lower().replace(" ", "-"))
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    
    while True:
        retV, frame = cam.read()
        abuabu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetector.detectMultiScale(abuabu, 1.2, 5)
        
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            namaFile1 = str(nim) + '_' + str(nama) + '_' + str(kelas) + '_' + str(ambilData_ori) +'.jpg'
            file_path1 = os.path.join(sub_dir, namaFile1)
            cv2.imwrite(file_path1, frame)

            # Histogram equalization
            namaFile2 = str(nim) + '_' + str(nama) + '_' + str(kelas) + '_' + str(ambilData_HE) +'.jpg'
            file_path2 = os.path.join(sub_dir, namaFile2)
            equalized = cv2.equalizeHist(abuabu)
            equalized_frame = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(file_path2, equalized_frame)

            # Filter berdasarkan tekstur (median blur)
            namaFile3 = str(nim) + '_' + str(nama) + '_' + str(kelas) + '_' + str(ambilData_texture) +'.jpg'
            file_path3 = os.path.join(sub_dir, namaFile3)
            filtered_frame = apply_texture_filter(frame)
            cv2.imwrite(file_path3, filtered_frame)
            
            # Flip image horizontal
            namaFile4 = str(nim) + '_' + str(nama) + '_' + str(kelas) + '_' + str(ambilData_flipped) +'.jpg'
            file_path4 = os.path.join(sub_dir, namaFile4)
            flipped_frame = cv2.flip(frame, 1)
            cv2.imwrite(file_path4, flipped_frame)

            ambilData_ori += 1
            ambilData_HE += 1
            ambilData_texture += 1
            ambilData_flipped += 1

            roiabuabu = equalized[y:y + h, x:x + w]
            roiwarna = equalized_frame[y:y + h, x:x + w]
            eyes = eyeDetector.detectMultiScale(roiabuabu)
            for (xe, ye, we, he) in eyes:
                cv2.rectangle(roiwarna, (xe, ye), (xe + we, ye + he), (0, 255, 255), 1)
        
        cv2.imshow('TEKAN Q UNTUK KELUAR', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # jika menekan tombol q akan berhenti
            break
        elif ambilData_ori > 150:
            break
    
    selesai1()
    cam.release()
    cv2.destroyAllWindows()

def trainingWajah():
    wajahDir = 'datawajah'
    latihDir = 'latihwajah'

    def getImageLabel(path):
        faceSamples = []
        faceIDs = []
        imageWidth = 0
        imageHeight = 0

        for root, dirs, files in os.walk(path):
            for file in files:
                imagePath = os.path.join(root, file)
                PILimg = Image.open(imagePath).convert('L')
                imgNum = np.array(PILimg, 'uint8')
                faceID = int(os.path.split(imagePath)[-1].split('_')[0])

                # Memeriksa dan menyamakan ukuran gambar
                if imageWidth == 0 and imageHeight == 0:
                    imageHeight, imageWidth = imgNum.shape
                else:
                    imgNum = cv2.resize(imgNum, (imageWidth, imageHeight))

                faces = faceDetector.detectMultiScale(imgNum)
                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        faceSamples.append(imgNum[y:y + h, x:x + w])
                        faceIDs.append(faceID)

        return faceSamples, faceIDs


    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    # faceRecognizer = cv2.face.EigenFaceRecognizer_create()
    # faceRecognizer = cv2.face.FisherFaceRecognizer_create()
    faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces, IDs = getImageLabel(wajahDir)
    faceRecognizer.train(faces, np.array(IDs))

    # Simpan model yang telah dilatih
    faceRecognizer.save(latihDir + '/training.xml')
    selesai2()

def markAttendance(name):
    with open("PRESENSI.csv", 'a+') as f:
        f.seek(0, os.SEEK_END)  # Posisi penulisan di akhir file
        if f.tell():  # Memeriksa apakah file tidak kosong
            f.write("\n")  # Menambahkan baris baru sebelum menulis data baru
        else:
            f.write("Nama,Kelas,NIM,Waktu,Mata_Kuliah\n")  # Menulis header jika file kosong

        yournim = entry2.get()
        yourclass = entry3.get()
        yourMata_Kuliah = entry4.get()
        
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        f.write(f'{name},{yourclass},{yournim},{dtString},{yourMata_Kuliah}')

def absensiWajah():
    wajahDir = 'datawajah'
    latihDir = 'latihwajah'
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    faceRecognizer.read(latihDir + '/training.xml')
    font = cv2.FONT_HERSHEY_SIMPLEX

    #id = 0
    yourname = entry1.get()
    names = []
    names.append(yourname)
    minWidth = 0.1 * cam.get(3)
    minHeight = 0.1 * cam.get(4)

    while True:
        retV, frame = cam.read()
        frame = cv2.flip(frame, 1)
        abuabu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetector.detectMultiScale(abuabu, 1.1, 4, minSize=(round(minWidth), round(minHeight)), )
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = faceRecognizer.predict(abuabu[y:y+h,x:x+w])
            # Print nilai confidence
            print("Confidence:", confidence)

            difference = round(130 - confidence)
            difference = min(difference, 100)  # Batasi nilai hasil selisih hingga maksimum 100

            if confidence <= 60:
                id = names[0]
                confidence = "  {0}%".format(difference)
            else:
                id = "Tidak Diketahui"
                confidence = "  {0}%".format(difference)

            cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(confidence), (x + 5, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow('TEKAN Q UNTUK KELUAR', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # jika menekan tombol q akan berhenti
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

# Menghilangkan border putih pada canvas
canvas = tk.Canvas(root, highlightthickness=0)
canvas.pack(fill="both", expand=True)

# canvas = tk.Canvas(root, width=1600, height=800)
# canvas.grid(columnspan=2, rowspan=8)
canvas.configure(bg="#FFA41B")

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
judul = tk.Label(root, text="SISTEM ABSENSI OTOMATIS", font=("Berlin Sans FB", 34, "bold"), bg="#362FD9", fg="#DDE6ED", relief="groove", bd=5)
canvas.create_window(800, 100, window=judul)
judul.place(relx=0.005, rely=0.01, relwidth=0.99, relheight=0.1)
######################################################################################################
# Load the image
gambar3 = Image.open("Logo Elektro.png")
gambar4 = Image.open("Logo UIN.png")

# Calculate the desired width and height
width, height = gambar3.size
desired_width3 = int(width * 0.086)
desired_height3 = int(height * 0.086)

width, height = gambar4.size
desired_width4 = int(width * 0.032)
desired_height4 = int(height * 0.032)

# Resize the image
gambar3 = gambar3.resize((desired_width3, desired_height3), Image.ANTIALIAS)
gambar4 = gambar4.resize((desired_width4, desired_height4), Image.ANTIALIAS)

# Convert the image to PhotoImage
photo3 = ImageTk.PhotoImage(gambar3)
photo4 = ImageTk.PhotoImage(gambar4)

# Create the label with the image
gambarlabel3 = tk.Label(root, image=photo3, bg="#FFA41B")
gambarlabel3.place(relx=0.1, rely=0.12, relwidth=0.059, relheight=0.13)
gambarlabel4 = tk.Label(root, image=photo4, bg="#FFA41B")
gambarlabel4.place(relx=0.03, rely=0.115, relwidth=0.05, relheight=0.15)

# gambar3 = PhotoImage(file="Logo Elektro.png")
# gambarlabel3 = Label(root, image=gambar3)
# gambarlabel3.place(relx=0.03, rely=0.14, relwidth=0.05, relheight=0.10)
######################################################################################################
# gambar4 = PhotoImage(file="Logo UIN.png")
# gambarlabel4 = Label(root, image=gambar4)
# gambarlabel4.place(relx=0.099, rely=0.14, relwidth=0.05, relheight=0.11)
######################################################################################################
#credit
# made = tk.Label(root, text=" ", font=("Calibri", 13, "italic"), bg="#362FD9", fg="black")
# canvas.create_window(700, 30, window=made)
# made.place(relx=0.7, rely=0.9, relwidth=0.3, relheight=0.1)

######################################################################################################
# fill data
fill_data = tk.Label(root, text="MOHON ISI DATA TERLEBIH DAHULU YA :)", font=("Berlin Sans FB", 20, "bold"), bg="#362FD9", fg="#DDE6ED")
canvas.create_window(0, 0, window=fill_data)
fill_data.place(relx=0.0, rely=0.9, relwidth=1, relheight=0.1)
################## Kolom Nama Mahasiswa #################################################################
# for entry data nama
entry1 = tk.Entry (root, font="Roboto")
canvas.create_window(457, 170, height=25, width=411, window=entry1)
entry1.place(relx=0.185, rely=0.35, relwidth=0.3, relheight=0.035)

label1 = tk.Label(root, text="Nama Mahasiswa", font=("Roboto", 20, "bold"), fg="white", bg="#FFA41B")
canvas.create_window(110,170, window=label1)
label1.place(relx=0.0, rely=0.341, relwidth=0.167, relheight=0.05)

################### Kolom NIM ############################################################################
# for entry data nim
entry2 = tk.Entry (root, font="Roboto")
canvas.create_window(457, 210, height=25, width=411, window=entry2)
entry2.place(relx=0.185, rely=0.4, relwidth=0.3, relheight=0.035)

label2 = tk.Label(root, text="NIM", font=("Roboto", 20, "bold"), fg="white", bg="#FFA41B")
canvas.create_window(60, 210, window=label2)
label2.place(relx=0.0, rely=0.395, relwidth=0.052, relheight=0.05)

##################### Kelas ################################################################################
# for entry data kelas
entry3 = tk.Entry (root, font="Roboto")
canvas.create_window(457, 250, height=25, width=411, window=entry3)
entry3.place(relx=0.185, rely=0.45, relwidth=0.3, relheight=0.035)

label3 = tk.Label(root, text="Kelas", font=("Roboto", 20, "bold"), fg="white", bg="#FFA41B")
canvas.create_window(65, 250, window=label3)
label3.place(relx=0.0, rely=0.445, relwidth=0.068, relheight=0.05)


########################################## Combo Box pemilihan mata kuliah ##################################


# label
ttk.Label(root, text="Mata Kuliah", background="#FFA41B", foreground="white",
          font=("Roboto", 20, "bold")).place(relx=0.0086, rely=0.496, relwidth=0.16, relheight=0.05)

# Combobox creation
entry4 = tk.StringVar()
classchoosen = ttk.Combobox(root, width=25, textvariable=entry4)

# Adding combobox drop down list
classchoosen['values'] = ('Sistem Tertanam dan Internet of Things IoT',
                          'Sistem Kecerdasan Buatan',
                          'Sistem Komunikasi',
                          'Praktikum Sistem Komunikasi',
                          'Pengolahan Citra Digital',
                          'Praktikum Pengolahan Citra Digital',
                          'Sistem Operasi',
                          'Etika Profesi dan Rekayasa',
                          'PPKN')

classchoosen.place(relx=0.185, rely=0.5, relwidth=0.3, relheight=0.035)
classchoosen.current()
#############################################################################################################
global intructions



# tombol untuk rekam data wajah
# for entry data kelas
label5 = tk.Label(root, text="Status Kehadiran", font=("Roboto", 20, "bold"), fg="white", bg="#FFA41B")
canvas.create_window(65, 250, window=label5)
label5.place(relx=0.00, rely=0.28, relwidth=0.168, relheight=0.05)
intructions = tk.Label(root, text="SELAMAT DATANG", font=("Calibri",16, 'bold'),fg="white",bg="#F86F03", relief="groove", bd=5)
canvas.create_window(370, 300, window=intructions)
intructions.place(relx=0.185, rely=0.28, relwidth=0.3, relheight=0.05)

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
                       command=trainingWajah)
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
frame3.place(relx=0.844, rely=0.20, relwidth=0.1, relheight=0.07)

frame4 = tk.Frame(root, bg="#65c6bb")
frame4.place(relx=0.79, rely=0.15, relwidth=0.2, relheight=0.07)

clock = tk.Label(frame3, fg="#27374D", bg="#FFA41B", width=70, height=2, font=('calibri', 22, ' bold '))
clock.pack(fill='both', expand=1)
tick()

datef = tk.Label(frame4, text = (day+" "+mont[month]+" "+year), fg="#27374D", bg="#FFA41B", width=70,
                 height=2,font=('calibri', 22, ' bold '))
datef.pack(fill='both', expand=1)

root.mainloop()
