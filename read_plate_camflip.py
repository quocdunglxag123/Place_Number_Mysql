
import cv2
import numpy as np
from lib_detection import load_model, detect_lp, im2single
from pathlib import Path
import mysql.connector
from datetime import date
import glob
import os.path

# Ham sap xep contour tu trai sang phai
def sort_contours(cnts):

    reverse = False
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

# Dinh nghia cac ky tu tren bien so
char_list =  '0123456789ABCDEFGHKLMNPRSTUVXYZ'

# Ham fine tune bien so, loai bo cac ki tu khong hop ly
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString

# Đường dẫn ảnh Test
#**************Edit********************

folder_path = r'C:\Users\quocd\OneDrive\Pictures\Camera Roll'
file_type = r'\*jpg'
files = glob.glob(folder_path + file_type)
if not files:
    print('Folder Empty!')
else:
    max_file = max(files, key=os.path.getctime)
    print(max_file) 
#**************End*********************
img_path = max_file

# Load model LP detection
wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)

# Đọc file ảnh đầu vào
img = cv2.imread(img_path)
Ivehicle = cv2.flip(img, 1)
#Ivehicle = cv2.imread(img_path)

# Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
Dmax = 608
Dmin = 288

# Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
side = int(ratio * Dmin)
bound_dim = min(side, Dmax)

_ , LpImg, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)


# Cau hinh tham so cho model SVM
digit_w = 30 # Kich thuoc ki tu
digit_h = 60 # Kich thuoc ki tu

model_svm = cv2.ml.SVM_load('svm.xml')

if (len(LpImg)):

    # Chuyen doi anh bien so
    LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))

    roi = LpImg[0]

    # Chuyen anh bien so ve gray
    gray = cv2.cvtColor( LpImg[0], cv2.COLOR_BGR2GRAY)


    # Ap dung threshold de phan tach so va nen
    binary = cv2.threshold(gray, 127, 255,
                         cv2.THRESH_BINARY_INV)[1]

    cv2.imshow("Anh bien so sau threshold", binary)
    cv2.waitKey()

    # Segment kí tự
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
    cont, _  = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


    plate_info = ""

    for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h/w
        if 1.5<=ratio<=3.5: # Chon cac contour dam bao ve ratio w/h
            if h/roi.shape[0]>=0.6: # Chon cac contour cao tu 60% bien so tro len

                # Ve khung chu nhat quanh so
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Tach so va predict
                curr_num = thre_mor[y:y+h,x:x+w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                curr_num = np.array(curr_num,dtype=np.float32)
                curr_num = curr_num.reshape(-1, digit_w * digit_h)

                # Dua vao model SVM
                result = model_svm.predict(curr_num)[1]
                result = int(result[0, 0])

                if result<=9: # Neu la so thi hien thi luon
                    result = str(result)
                else: #Neu la chu thi chuyen bang ASCII
                    result = chr(result)

                plate_info +=result

    cv2.imshow("Cac contour tim duoc", roi)
    cv2.waitKey()

    #**************EDIT***********
    # Connect to sql_server
    mydb = mysql.connector.connect(
      host = "localhost",
      user = "root",
      password = "quocdung",
      database = "numberplate"
    ) 
    # selecting Data
    mycursorGet = mydb.cursor()
    queryGet = "SELECT * FROM Infor_Vehicel_in where number_plate= %s"
    valGet = (plate_info,)
    mycursorGet.execute(queryGet,valGet)
    myresultGet = mycursorGet.fetchall()
    if not myresultGet:
         # Viet bien so len anh
        cv2.putText(Ivehicle,fine_tune(plate_info),(50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), lineType=cv2.LINE_AA)
        # Hien thi anh
        print("Bien so=", plate_info)
        cv2.imshow("Hinh anh output",Ivehicle)
        cv2.waitKey()
        # Insert Number Plate
        mycursorInsert = mydb.cursor()
        datein= date.today()
        formatted_date = datein.strftime('%Y-%m-%d')
        queryInsert = "INSERT INTO Infor_Vehicel_in (number_plate, time_in) VALUES (%s, %s)"
        valInsert = (plate_info, formatted_date)
        mycursorInsert.execute(queryInsert, valInsert)
        mydb.commit()
        print(mycursorInsert.rowcount, "details inserted")
    else:
        for x in myresultGet:
            dateInDB= x[2]
        datestay= date.today() - dateInDB
        price = datestay.days *10000 + 5000
        # Viet bien so len anh
        price_numberplate= str(price) + 'VND'
        cv2.putText(Ivehicle,fine_tune(plate_info),(50, 100), cv2.FONT_HERSHEY_PLAIN, 5.0, (0, 0, 255), lineType=cv2.LINE_AA)
        cv2.putText(Ivehicle,fine_tune(price_numberplate),(50, 200), cv2.FONT_HERSHEY_PLAIN, 5.0, (0, 0, 255), lineType=cv2.LINE_AA)
        # Hien thi anh
        price_numberplate= 'Bien So Xe: '+ plate_info + ' Gui Xe Trong: ' + str(datestay.days) + ' Ngay ' + ' Gia Tien: ' + str(price) 
        print(price_numberplate)
        cv2.imshow("Hinh anh output",Ivehicle)
        cv2.waitKey()
        #Delete Vehicel Check Out
        mycursorDelete = mydb.cursor()
        queryDelete = "DELETE FROM Infor_Vehicel_in where number_plate= %s"
        valDelete = (plate_info,)
        mycursorDelete.execute(queryDelete,valDelete)
        mydb.commit() 
        
        # disconnecting from server
        mydb.close()
       
    
    #**************END**********
#Delete Image
if os.path.exists(img_path):
    os.remove(img_path)
else:
    print("The file does not exist")
cv2.destroyAllWindows()
"""
    price_numberplate= 'Bien So Xe: '+ plate_info + ' Gui Xe Trong: ' + str(datestay.days) + ' Ngay ' + ' Gia Tien: ' + str(price) 
    # Viet bien so len anh
    cv2.putText(Ivehicle,fine_tune(plate_info),(50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), lineType=cv2.LINE_AA)

    # Hien thi anh
    print("Bien so=", plate_info)
    cv2.imshow("Hinh anh output",Ivehicle)
    cv2.waitKey()
"""
