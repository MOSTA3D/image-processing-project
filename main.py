# # # This is a sample Python script.
# #
# # # Press Shift+F10 to execute it or replace it with your code.
# # # Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# #
# # #
# # # def print_hi(name):
# # #     # Use a breakpoint in the code line below to debug your script.
# # #     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
# # #
# # #
# # # # Press the green button in the gutter to run the script.
# # # if __name__ == '__main__':
# # #     print_hi('PyCharm')
# # #     print(__name__)
# # """
# # bitmap = "error" , "hourglass" , 'info" , "grey50" , "questhead","warning","question"
# # """
# # from tkinter import *
# # window = Tk()
# # window.geometry("500x500+150+300")
# # window.title("hello tkinter")
# #
# # label = Label(text=" hello Tkinter", fg="#f00", bg="#00f").grid(row=0, column=0)
# # #grid is using row and coluns
# # button2 = Button(text="some other button", state=DISABLED, fg="red", bg="blue").grid(row=1, column=1)
# # #place is using x and y
# # button = Button(text="hello tkinter").place(x=300, y=150)
# # from tkinter import *
# # window = Tk()
# # window.title("hello tkinter")
# # window.geometry("500x500+150+150")
#
# #
# # class MyFirstGui:
# #     def __init__(self,master):
# #         self.master = master
# #         master.title("GUI")
# #         self.label = Label(master,text="hello tkinter")
# #         self.label.pack()
# #         self.greet_button = Button(master, text="welcome", command=self.greet)
# #         self.greet_button.pack()
# #         self.button2 = Button(text="quit", command=self.greet_button.quit).pack()
# #         # self.logo = PhotoImage(file="somephoto.png")
# #         # self.button.config(image = self.logo)
# #
# #     def greet(self):
# #         self.label2 = Label(self.master, text="this is the text 2")
# #         self.label2.pack()
# #
# #
# # my_gui = MyFirstGui(window)
# # window.mainloop()
# # from tkinter import *
# # from tkinter import ttk
# # window = Tk()
# # some = ""
# # window.title("some title")
# # window.geometry("500x500+500+100")
# # frame = Frame(window,height=100, width=100, relief=RIDGE, padx=100,pady = 100)
# # frame.pack()
# # frame3 = LabelFrame(window, text="this is frame",padx = 5,pady=10)
# # frame3.pack()
# # frame2 = ttk.Frame(window,width=200, height=50, relief=RIDGE).pack()
# # button = Button(frame3, text="click",relief = RIDGE).pack()
# # window.mainloop()
# from tkinter import *
# window = Tk()
# window.title("title")
# window.geometry("500x500+450+100")
# Frame(window, height = 250,width=100,bg="#333").pack()
# window.configure(pady=200)
# window.mainloop()
import base64
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2 as cv
import os
import numpy as np
from random import random
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


root = Tk()
root.configure(bg="#333")
win_width = 850
win_height = 650
left_container_width = 0.6 * win_width
# myframe1 = LabelFrame(root,text="this is my frame")
# myframe1.pack(side=BOTTOM,padx=15,pady=15)
# mylabel1 = Label(root)
# mylabel1.pack()
#
# button1 = Button(myframe1,text="Exit",command=lambda:exit())
# button1.pack(side=LEFT,padx=10)
#
# button2 = Button(myframe1,text="Browe Image",ommand=showimage)
# button2.pack(side=LEFT)
ten = 10

root.option_add( "*font", "Arial 10" )
root.option_add("*Background","#333")
root.option_add("*Foreground","#fff")
root.option_add("*Button*BorderWidth",2)
root.option_add("*Button*Background","#444")

# right container that contain the results
right_container_width = win_width - left_container_width
right_container = Frame(root, width=right_container_width, height=win_height+10)
right_container.grid(row=0, column=1)


original_image = LabelFrame(right_container, width=right_container_width,height=win_height/3+7,text="Original image")
original_image.grid(row=0)
original_image_label = Label(original_image, width=int(right_container_width)-10, height=int(win_height/3)-20, image="")
original_image_label.place(x=2, y=0)


after_noise = LabelFrame(right_container, width= right_container_width,height=win_height/3+7,text="after noise added")
after_noise.grid(row=1)
after_noise_label = Label(after_noise, width=int(right_container_width)-10, height=int(win_height/3)-20)
after_noise_label.place(x=2, y=0)


result = LabelFrame(right_container, text="Result", width=right_container_width, height=win_height/3+7)
result.grid(row=2)
result_label = Label(result, width=int(right_container_width)-10, height=int(win_height/3)-20)
result_label.place(x=3, y=3)

img = ""
result_img = ""
# my functions


def showimage(color):
    global original_image_label
    global img
    global result_img
    # original_image_label.destroy()
    image_file = filedialog.askopenfile(initialdir=os.getcwd(),title="Select Image File.",filetypes =(("PNG file","*.png"),("JPG file","*.jpg"),("ALL FILES","*.*")))
    img = Image.open(image_file.name)
    img.thumbnail((350, 400))
    # img=img.decode("latin-1").encode("utf-8")
    # img = base64.encodebytes(img)
    # cv_img = cv.imread(image_file.name)

    ph_img = np.asarray(img)
    if color == 2:
        ph_img = cv.cvtColor(ph_img, cv.COLOR_BGR2GRAY)
    result_img = ph_img
    ph_img = Image.fromarray(ph_img)
    ph_img = ImageTk.PhotoImage(ph_img)
    print(color)
    original_image_label.configure(image=ph_img)
    original_image_label.image = ph_img

    # original_image_label.configure(text="some text", bg="#fff")
    # root.mainloop()


def change_img_color(color):
    global img
    global original_image_label
    global result_img
    # print(np.asarray(img))
    limg = np.asarray(img)
    global original_image
    if color == 2:
        limg = cv.cvtColor(limg, cv.COLOR_BGR2GRAY)
        original_image.configure(text="Grey scale image")
    result_img = limg
    limg = Image.fromarray(limg)
    limg = ImageTk.PhotoImage(limg)

    original_image_label.configure(image=limg)
    original_image_label.image = limg


    print("********************")
    root.mainloop()


def add_noise(noise_typ,local_color):
    global after_noise_label
    local_img = np.asarray(img)
    noisy = 0
    global after_noise
    if local_color == 2:
        local_img = cv.cvtColor(local_img, cv.COLOR_BGR2GRAY)

    if noise_typ == 2:
        row,col,ch= local_img.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = local_img + gauss

    elif noise_typ == 1:
        row,col,ch = local_img.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(local_img)
        # Salt mode
        num_salt = np.ceil(amount * local_img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in local_img.shape]
        out[coords] = 1

        # Pepper mode

        num_pepper = np.ceil(amount* local_img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in local_img.shape]
        out[coords] = 0

    elif noise_typ == 3:
        vals = len(np.unique(local_img))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(local_img * vals) / float(vals)

    local_img = Image.fromarray(local_img)
    local_img = ImageTk.PhotoImage(local_img)

    after_noise_label.configure(image=local_img)
    after_noise_label.image = local_img
    after_noise.configure(text="After noise added")

print("**********", int(random()*256) - 1, "99999999999")


def brt_adjst(color):
    global result_label
    global img
    global result_img
    global result
    local_img = np.asarray(img)
    print(local_img.shape)
    alpha = 1.0  # Simple contrast control
    beta = 0     # Simple brightness control
    random_brightness = random() * 256
    if color == 2:
        local_img = cv.cvtColor(local_img, cv.COLOR_BGR2GRAY)
        # local_img = local_img.reshape([1,local_img.shape[0],local_img.reshape[1]])
        # local_img = local_img[..., np.newaxis]
        print(local_img.shape)
        for y in range(local_img.shape[0]):
            for x in range(local_img.shape[1]):
                local_img[y, x] = int((local_img[y,x] + random_brightness)/2)
    elif color == 1:
        for y in range(local_img.shape[0]):
            for x in range(local_img.shape[1]):
                for c in range(local_img.shape[2]):
                    local_img[y, x, c] = (local_img[y, x, c] + random_brightness)
                    # print(new_image[y, x, c])

    # new_image = np.zeros(local_img.shape, local_img.dtype)
    print("*************")
    print(local_img.shape)
    print("**************")
    result_img = local_img
    local_img = Image.fromarray(local_img)
    local_img = ImageTk.PhotoImage(local_img)

    result_label.configure(image=local_img)
    result_label.image = local_img
    result.configure(text="Brightness adjustment")

def cntrst_adjst(color):
    global result_label
    global img
    global result_img
    global result
    local_img = np.asarray(img)
    random_contrast = random()
    if color == 2:
        local_img = cv.cvtColor(local_img,cv.COLOR_BGR2GRAY)
        for y in range(local_img.shape[0]):
            for x in range(local_img.shape[1]):
                local_img[y, x] = int(local_img[y, x] * random_contrast) % 256
    elif color == 1:
        for y in range(local_img.shape[0]):
            for x in range(local_img.shape[1]):
                for c in range(local_img.shape[2]):
                    local_img[y, x, c] = int(local_img[y, x, c] * random_contrast) % 256

    result_img = local_img
    local_img = Image.fromarray(local_img)
    local_img = ImageTk.PhotoImage(local_img)
    result_label.configure(image=local_img)
    result_label.image = local_img
    result.configure(text="Contrast adjustment")


def show_histogram(clr):
    global after_noise_label
    global result_img
    global after_noise
    local_img = result_img
    print(local_img.shape)
    figure = Figure(figsize=(3.3,2), dpi=100)
    som = figure.add_subplot(111)
    histogram = cv.calcHist([local_img],[0],None,[256],[0,256])
    print("this histogram is = ")
    print(histogram)
    som.plot(histogram)
    chart_type = FigureCanvasTkAgg(figure, after_noise_label)
    chart_type.get_tk_widget().place(x=2,y=2)
    plt.plot(legend=True, ax=som)
    # canvas.show()
    # canvas.get_tk_widget().place(x=3,y=2)
    after_noise.configure(text="Histogram")


def hst_eqlz(clr):
    global result_label
    global result_img
    global img
    global result
    local_img = np.asarray(img)
    local_img = cv.cvtColor(local_img, cv.COLOR_BGR2GRAY)
    local_img = cv.equalizeHist(local_img)
    result_img = local_img
    local_img = Image.fromarray(local_img)
    local_img = ImageTk.PhotoImage(local_img)
    result_label.configure(image=local_img)
    result_label.image = local_img
    result.configure(text="Histogram equalization")

def lwpssfltr(clr):
    global result_label
    global result_img
    global img
    global result
    local_img = np.asarray(img)
    if clr == 2:
        local_img = cv.cvtColor(local_img, cv.COLOR_BGR2GRAY)
    local_img = cv.blur(local_img, (7,7), cv.BORDER_DEFAULT)
    result_img = local_img
    local_img = Image.fromarray(local_img)
    local_img = ImageTk.PhotoImage(local_img)
    result_label.configure(image=local_img)
    result_label.image = local_img
    result.configure(text="Low pass filter")


def hghpssfltr(clr):
    global img
    global result_label
    global result_img
    global result
    local_img = np.asarray(img)
    if clr == 2:
        local_img = cv.cvtColor(local_img, cv.COLOR_BGR2GRAY)
    print("the shape of highpass is")
    print(local_img.shape)
    result_img = local_img - cv.GaussianBlur(local_img, (0,0), 3) + 127

    local_img = Image.fromarray(result_img)
    local_img = ImageTk.PhotoImage(local_img)
    result_label.configure(image=local_img)
    result_label.image = local_img
    result.configure(text="High pass filter")


def mdn_fltr(clr):
    global img
    global result_label
    global result_img
    global result

    local_img = np.asarray(img)
    if clr == 2:
        local_img = cv.cvtColor(local_img, cv.COLOR_BGR2GRAY)
    local_img = cv.medianBlur(local_img, 5)
    result_img = local_img
    local_img = Image.fromarray(local_img)
    local_img = ImageTk.PhotoImage(local_img)
    result_label.configure(image=local_img)
    result_label.image = local_img
    result.configure(text="Median filter")


def avg_fltr(clr):
    global img
    global result_label
    global result_img
    global result
    local_img = np.asarray(img)
    if clr == 2:
        local_img = cv.cvtColor(local_img, cv.COLOR_BGR2GRAY)
    local_img = cv.blur(local_img, (5,5))
    result_img = local_img
    local_img = Image.fromarray(local_img)
    local_img = ImageTk.PhotoImage(local_img)
    result_label.configure(image=local_img)
    result_label.image = local_img
    result.configure(text="average filter")


def hgh_ln(clr):
    global img
    global result_label
    global result_img
    local_img = np.asarray(img)
    # copy = local_img
    local_img = cv.cvtColor(local_img, cv.COLOR_BGR2GRAY)
    copy = np.zeros(local_img.shape, local_img.dtype)
    lines = cv.HoughLinesP(cv.Canny(local_img, 50, 150,apertureSize=3), 1, np.pi / 180, 50,200)
    print("the first rho", lines[0][0][0])

    for line in lines:
        rho, theta, _, _ = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0+1000*(-b))
        y1 = int(y0+1000*a)
        x2 = int(x0-1000*(-b))
        y2 = int(y0-1000*a)
        print("the line is ","***********",[x1,y1,x2,y2], "**************8")
        cv.line(copy, (x1,y1), (x2,y2), (0,0,255), 2)
    print(copy)
    local_img = copy
    result_img = local_img
    print("the line hough shape is ")
    print(local_img.shape)
    local_img = Image.fromarray(local_img)
    local_img = ImageTk.PhotoImage(local_img)
    result_label.configure(image=local_img)
    result_label.image = local_img


def hgh_crl():
    global img
    global result_label
    global result_img
    local_img = np.asarray(img)
    copy = local_img
    local_img = cv.cvtColor(local_img, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(cv.medianBlur(local_img, 5), cv.HOUGH_GRADIENT, 1,20, param1=50, param2=30, minRadius=0, maxRadius=0)
    detected_circles = np.uint16(np.around(circles))
    print("the circle one is : ",detected_circles[0])
    for (x,y,r) in detected_circles[0,:]:
        cv.circle(copy, (x,y), r, (0, 255, 255), 3)
        cv.circle(copy, (x,y), 2, (0, 255, 255), 3)
    result_img = copy
    local_img = copy
    local_img = Image.fromarray(local_img)
    local_img = ImageTk.PhotoImage(local_img)
    result_label.configure(image=local_img)
    result_label.image = local_img


def add_fltr(clr, fltr):
    global img
    global result_label
    global result_img
    global result

    local_img = np.asarray(img)
    local_img = cv.cvtColor(local_img, cv.COLOR_BGR2GRAY)
    if clr == 2:
        local_img = cv.cvtColor(local_img, cv.COLOR_BGR2GRAY)

    if fltr == "Laplas filter":
        local_img = cv.Laplacian(local_img, cv.CV_16S, ksize=3)
    elif fltr == "Gussian filter":
        local_img = cv.GaussianBlur(local_img, (3, 3), 0)
    elif fltr == "Horiz Sobel":
        local_img = cv.Sobel(local_img, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    elif fltr == "Vert Sobel":
        local_img = cv.Sobel(local_img, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    elif fltr == "Vert Prewitt":
        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        local_img = cv.filter2D(cv.GaussianBlur(local_img, (3, 3), 0), -1, kernelx)
    elif fltr == "Horiz perwitt":
        kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        local_img = cv.filter2D(cv.GaussianBlur(local_img, (3,3), 0), -1, kernely)
    elif fltr == "Lap of Gau(log)":
        local_img = cv.Laplacian(cv.GaussianBlur(local_img, (3, 3), 0), cv.CV_16S, ksize=3)
    elif fltr == "Canny method":
        local_img = cv.Canny(cv.GaussianBlur(local_img, (3, 3), 0), 50, 150,apertureSize=3)
    elif fltr == "Zero Cross":
        LoG = cv.Laplacian(local_img, cv.CV_16S)
        minLoG = cv.morphologyEx(LoG, cv.MORPH_ERODE, np.ones((3,3)))
        maxLoG = cv.morphologyEx(LoG, cv.MORPH_DILATE, np.ones((3,3)))
        local_img = np.logical_or(np.logical_and(minLoG < 0,  LoG > 0), np.logical_and(maxLoG > 0, LoG < 0))
    # elif fltr == "Ticken":
    #     x= 5-5
    # elif fltr == "Skeleton":
    #     size = np.size(img)
    #     skel = np.zeros(local_img.shape, np.uint8)
    #     blur= cv.GaussianBlur(local_img,(5,5),0)
    #     ret,thrs=cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    #     inv=cv.bitwise_not(thrs)
    #     element = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    #     done = False
    #
    #     while( not done):
    #         eroded = cv.erode(inv,element)
    #         temp = cv.dilate(eroded,element)
    #         temp = cv.subtract(inv,temp)
    #         skel = cv.bitwise_or(skel,temp)
    #         inv = eroded.copy()
    #     zeros = size - cv.countNonZero(inv)
    #     if zeros==size:
    #         done = True
    #     local_img = skel
    #
    # elif fltr == "Thinning":
    #     # local_img = cv.ximgproc.thinning(local_img)
    #     kernel = cv.getStructuringElement(cv.MORPH_CROSS,(3,3))
    #     # Create an empty output image to hold values
    #     thin = np.zeros(local_img.shape,dtype='uint8')
    #     img1 = local_img
    #     # Loop until erosion leads to an empty set
    #     while (cv.countNonZero(img1)!=0):
    #         # Erosion
    #         erode = cv.erode(img1,kernel)
    #         # Opening on eroded image
    #         opening = cv.morphologyEx(erode,cv.MORPH_OPEN,kernel)
    #         # Subtract these two
    #         subset = erode - opening
    #         # Union of all previous sets
    #         thin = cv.bitwise_or(subset,thin)
    #         # Set the eroded image for next iteration
    #         img1 = erode.copy()
    #         local_img = img1

    result_img = local_img
    local_img = Image.fromarray(local_img)
    local_img = ImageTk.PhotoImage(local_img)
    result_label.configure(image=local_img)
    result_label.image = local_img
    result.configure(text=fltr)

""" Laplas filter   
   Gussian filter
  Vert Sobel     
  Horiz Sobel    
    Vert Prewitt 
  Horiz perwitt  
 Lap of Gau(log) 
  Canny method   
   Zero Cross    
     Ticken      
      Skeleton   
     Thinning    """


def mrph_fltrs(clr, type):
    global img
    global result_label
    global result_img
    global result
    local_img = np.asarray(img)
    kernel = np.ones((5,5), 'uint8')
    if clr == 2:
        local_img = cv.cvtColor(local_img, cv.COLOR_BGR2GRAY)
    if type == "Dilation filter":
        local_img = cv.dilate(local_img, kernel, iterations=1)
    elif type == "Erosion filter":
        local_img = cv.erode(local_img, kernel, iterations = 1)
    elif type == "Open filter":
        local_img = cv.morphologyEx(local_img, cv.MORPH_OPEN, kernel)
    elif type == "Close filter":
        local_img = cv.morphologyEx(local_img, cv.MORPH_CLOSE, kernel)

    result_img = local_img
    local_img = Image.fromarray(local_img)
    local_img = ImageTk.PhotoImage(local_img)
    result_label.configure(image=local_img)
    result_label.image = local_img
    result.configure(text=type)


# def ersn_fltr(clr):
#     global img
#     global result_label
#     global result_img
#     global result
#     local_img = np.asarray(img)
#     if clr == 2:
#         local_img = cv.cvtColor(local_img, cv.COLOR_BGR2GRAY)
#     result_img = local_img
#     local_img = Image.fromarray(local_img)
#     local_img = ImageTk.PhotoImage(local_img)
#     result_label.configure(image=local_img)
#     result_label.image = local_img
#     result.configure(text="dilation filter")


def saveimage():
    # saved = filedialog.asksaveasfile(defaultextension=".jpg",filetypes=[("PNG file","*.png"),("JPG file","*.jpg")])

    saved = filedialog.asksaveasfile(mode="w", initialdir=os.getcwd(),filetypes=(("PNG file", "*.png"), ("JPG file", "*.jpg"), ("ALL FILES", "*.*")))
    image = Image.fromarray(result_img)
    image.save(saved.name)
    print(saved)


# left container that contain the functionality of the program
root.grid_propagate(0)

left_container = LabelFrame(root, text="this is the container", width=left_container_width)

left_container.grid(row=0,column=0,pady=10)

top_container = Frame(left_container, width=left_container_width,height=130)
top_container.grid(row=0,column=0)
load_image = LabelFrame(top_container, text="Load Image",width=left_container_width/3-20,height=120)
load_image.place(x=10, y=10)
load_button = Button(load_image,text="load image", command=lambda:showimage(color.get()))
load_button.place(x=30, y=35)
print(load_button.winfo_width())

convert = LabelFrame(top_container,width=left_container_width/3-20, height=120, text="Convert grey")
convert.place(x=left_container_width/3, y=10)

color = IntVar()
noise = IntVar()
edge = StringVar()
color.set("2")
noise.set("1")
edge.set("lablas")


default_color = Radiobutton(convert, text="Default color", variable=color, value=1, command=lambda: change_img_color(color.get()))
default_color.place(x=25, y=10)
grey_color = Radiobutton(convert, text="  grey color   ", variable=color, value=2, command=lambda: change_img_color(color.get()))
grey_color.place(x=25, y=60)

add_note = LabelFrame(top_container, text="Add note", width=left_container_width/3, height=120)
add_note.place(x=2*left_container_width/3,y=10)
Radiobutton(add_note, text="salt & pepper noise ", variable=noise, value=1, command=lambda: add_noise(noise.get(), color.get())).place(x=12, y=5)
Radiobutton(add_note, text="  Gaussian noise     ", variable=noise, value=2, command=lambda: add_noise(noise.get(), color.get())).place(x=12, y=37)
Radiobutton(add_note, text="   Poission noise     ", variable=noise, value=3, command=lambda: add_noise(noise.get(), color.get())).place(x=12, y=69)
# # end left-conteiner top
# # start 2nd frame in left-container
#
point_transform = LabelFrame(left_container, text="Point Transform Op's", width=left_container_width,height=150)
point_transform.grid(row=1)

br_button = Button(point_transform, text="Brightness adjustment", width=20, command=lambda: brt_adjst(color.get()))
br_button.place(x=60,y=30)
cnt_button=Button(point_transform, text="Contrast adjustment", width=20, command=lambda: cntrst_adjst(color.get()))
cnt_button.place(x=265,y=30)
hst_button = Button(point_transform, text="Histogram",width=20, command=lambda: show_histogram(color.get()))
hst_button.place(x=60,y=75)
hst_eq_button = Button(point_transform, text="Histogram equalization", width=20, command=lambda: hst_eqlz(color.get))
hst_eq_button.place(x=265, y=75)

local_transform_ops = LabelFrame(left_container,text="Local Transform Ops",width=left_container_width,height=400)
local_transform_ops.grid(row=2)
bt_container = Frame(local_transform_ops,height=150,width=120)
bt_container.grid(row=0,column=0, padx=10,pady=10)
low_pass_filter = Button(bt_container,text="  Low Pass filter ", command=lambda: lwpssfltr(color.get()))
low_pass_filter.grid(row=0,pady=5)
high_pass_filter = Button(bt_container,text="  High pass filter ", command=lambda: hghpssfltr(color.get()))
high_pass_filter.grid(row=1, pady=5)
median_filter = Button(bt_container,text=" Median filtering  ", command=lambda: mdn_fltr(color.get()))
median_filter.grid(row=2, pady=5)
average_filter = Button(bt_container,text="  Averaging filter  ", command=lambda: avg_fltr(color.get()))
average_filter.grid(row=3, pady=5)
fltrs_container = LabelFrame(local_transform_ops, text="Edge detection filters", height=150, width=240)
fltrs_container.grid(row=0,column=1)
Radiobutton(fltrs_container,text="   Laplas filter    ",variable=edge,value="Laplas filter", command=lambda:add_fltr(color, edge.get())).grid(row=0,column=0,pady=5)
Radiobutton(fltrs_container,text="   Gussian filter    ",variable=edge,value="Gussian filter", command=lambda:add_fltr(color, edge.get())).grid(row=0,column=1,pady=5)
Radiobutton(fltrs_container,text="   Vert Sobel     ",variable=edge,value="Vert Sobel", command=lambda:add_fltr(color, edge.get())).grid(row=0,column=2,pady=5)
Radiobutton(fltrs_container,text="  Horiz Sobel     ",variable=edge,value="Horiz Sobel", command=lambda:add_fltr(color, edge.get())).grid(row=1,column=0,pady=5)
Radiobutton(fltrs_container,text="    Vert Prewitt     ",variable=edge,value="Vert Prewitt", command=lambda:add_fltr(color, edge.get())).grid(row=1,column=1,pady=5)
Radiobutton(fltrs_container,text="  Horiz perwitt   ",variable=edge,value="Horiz perwitt", command=lambda:add_fltr(color, edge.get())).grid(row=1,column=2,pady=5)
Radiobutton(fltrs_container,text=" Lap of Gau(log) ",variable=edge,value="Lap of Gau(log)", command=lambda:add_fltr(color, edge.get())).grid(row=2,column=0,pady=5)
Radiobutton(fltrs_container,text="  Canny method   ",variable=edge,value="Canny method", command=lambda:add_fltr(color, edge.get())).grid(row=2,column=1,pady=5)
Radiobutton(fltrs_container,text="   Zero Cross    ",variable=edge,value="Zero Cross ", command=lambda:add_fltr(color, edge.get())).grid(row=2,column=2,pady=5)
Radiobutton(fltrs_container,text="     Ticken         ",variable=edge,value="Ticken", command=lambda:add_fltr(color, edge.get())).grid(row=3,column=0,pady=5)
Radiobutton(fltrs_container,text="       Skeleton      ",variable=edge,value="Skeleton", command=lambda:add_fltr(color, edge.get())).grid(row=3,column=1,pady=5)
Radiobutton(fltrs_container,text="     Thinning      ",variable=edge,value="Thinning", command=lambda:add_fltr(color, edge.get())).grid(row=3,column=2,pady=5)




glomo_frame = Frame(left_container, width=left_container_width, height=150)
glomo_frame.grid(row=3)


global_transform = LabelFrame(glomo_frame, text="Global transform Ops", width=left_container_width/2+50,height=120)
global_transform.place(x=0,y=10)
Button(global_transform, text="  Line detection using hough transform  ", command=lambda: hgh_ln(color.get())).place(x=20,y=12)
Button(global_transform, text="Circles detection using hough transform", command=hgh_crl).place(x=20, y=55)


morpho_ops = LabelFrame(glomo_frame,text="Morphological Ops", width=left_container_width/2-50,height=120)
morpho_ops.place(x=left_container_width/2+50, y=10)
# morpho_btn_frame = Frame(morpho_ops)
# morpho_btn_frame.place(x=3,y=3)
Button(morpho_ops, text=" Dilation ", command=lambda: mrph_fltrs(color.get(), "Dilation filter")).place(x=11,y=7)
Button(morpho_ops, text=" Erosion  ", command=lambda: mrph_fltrs(color.get(), "Erosion filter")).place(x=110,y=7)
Button(morpho_ops, text="  Close   ", command=lambda: mrph_fltrs(color.get(), "Close filter")).place(x=11,y=60)
Button(morpho_ops, text="   Open   ", command=lambda: mrph_fltrs(color.get(), "Open filter")).place(x=110,y=60)


#
# Label(morpho_ops, text="Choose type of Kernel: ").grid(row=0,column=1)

under_btn_frame = Frame(left_container, width=left_container_width)
under_btn_frame.grid(row=4)
save_result = Button(under_btn_frame, text="save result image", command=saveimage)
save_result.grid(row=1, column=0)

quite = Button(under_btn_frame, text="Exit", command=lambda: exit())
quite.grid(row=1, column=1)





#right_top.grid(row=0,column=1)
root.title("some title")
root.geometry("900x700")
root.mainloop()



# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
