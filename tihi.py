import cv2
from scipy import ndimage
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
import math
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import *
from skimage import color

cc = -1

def nextId():
    global cc
    cc += 1
    return cc
def convertIMG(img):
   #izdvajanje belih piksela
    img_BW=img==1.0
    return img_BW
def inRange(r, item, items):
    retVal = []
    for obj in items:
        mdist = distance(item['center'], obj['center'])
        if(mdist<r):
            retVal.append(obj)
    return retVal

def my_rgb2gray(img_rgb):
    img_gray = np.ndarray((img_rgb.shape[0], img_rgb.shape[1]))
    img_gray = 0.5*img_rgb[:, :, 0] + 0.0*img_rgb[:, :, 1] + 0.5*img_rgb[:, :, 2]
    img_gray = img_gray.astype('uint8')
    return img_gray



videoName="Videos/video-9.avi"

#izdvajanje slike prvog frejma
def findLineParams(videoName):
    cap = cv2.VideoCapture(videoName)
    i=0
    access = True
    gray="grayFrame"
    frame1="frame"
    if access:
        access=False
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame1=frame

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            else:
                break
        
        print("Slika prvog frejma uspesno uhvacena")
       
        cap.release()
        cv2.destroyAllWindows()

    return houghTransformtion(frame1,gray)




def houghTransformtion(frame,grayImg):
   # img = cv2.imread('dave.jpg')
  #  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayImg,50,150,apertureSize = 3)

    minx,miny,maxx,maxy=advancedHoughTransformation(frame,edges,600,10)
   # print("minx={minx} miny={miny} maxx={maxx} maxy={maxy}".format(minx=minx, miny=miny, maxx=maxx, maxy=maxy))
  #  roughHoughTransformation(frame,edges)
    cv2.imwrite('linija.jpg',frame)
    return minx,miny, maxx,maxy


def advancedHoughTransformation(frame,edges,minLineLength,maxLineGap):
   # minLineLength = 100
   # maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength, maxLineGap)

    minx=0
    miny=0
    maxy=0
    maxx=0
    for x1, y1, x2, y2 in lines[0]:
        minx=x1
        miny=y1
        maxx=x2
        maxy=y2


    for i in  range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
           # print("x1= {x1} y1={y1} x2={x2}   y2={y2}.".format(x1=x1, x2=x2, y1=y1, y2=y2))
            if x1<minx :
                minx=x1
                miny=y1
            if x2>maxx:
                maxy=y2
                maxx=x2

    cv2.line(frame, (minx,miny), (maxx, maxy), (0, 255, 0), 2)
    return minx,miny,maxx,maxy
      # cv2.line(frame, (399, 118), (429, 96), (0, 255, 0), 2)
      #  cv2.line(frame, (250, 60), (500, 130), (0, 255, 0), 2)
   # cv2.imwrite('houghlines5.jpg', img)

def roughHoughTransformation(frame,edges):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
      #  print("x1= {x1} y1={y1} x2={x2}   y2={y1}.".format(x1=x1, x2=x2, y1=y1, y2=y2))
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
mnist = fetch_mldata('MNIST original')
def findMinXAndMinY(regions):
    n = len(regions)
    x_min = np.empty(n)
    x_max = np.empty(n)
    y_min = np.empty(n)
    y_max = np.empty(n)
    j = 0
    for region1 in regions:
        bbox1 = region1.bbox
        x_min[j] = bbox1[0]
        x_max[j] = bbox1[2]
        y_min[j] = bbox1[1]
        y_max[j] = bbox1[3]
        j = j + 1
    minX = np.amin(x_min)
    minY = np.amin(y_min)
    maxX = np.amax(x_max)
    maxY = np.amax(y_max)
    return (minX, maxX, minY, maxY)


def findRegions(img):
    img_1 = (color.rgb2gray(img) * 255).astype('uint8')
    imgBW = img_1 > 0

    print imgBW.shape
    label_img = label(imgBW)
    regions = regionprops(label_img)
    validRegions = []
    digitImages = []
    imageBoundary = []

    for region in regions:
        bbox = region.bbox

        print "Pozicije: " + format(bbox[0]) + " " + format(bbox[1]) + " " + format(bbox[2]) + " " + format(bbox[3])

        if (bbox[2] - bbox[0] > 10 or bbox[3] - bbox[1]>10):
            if (bbox[0] - 10 > 0 and bbox[0] + 10 < 784 and bbox[1] - 10 > 0 and bbox[1] + 10 < 784):
                label_img1 = label(imgBW[bbox[0] - 10:bbox[2] + 10, bbox[1] - 10:bbox[3] + 10])
                regions1 = regionprops(label_img1)
                imageDigit2 = img[bbox[0] - 10:bbox[2] + 10, bbox[1] - 10:bbox[3] + 10]
               # height=imageDigit2.shape[1]
                #width=imageDigit2.shape[0]

                result = findMinXAndMinY(regions1)
                print "result[0] " + format(result[0]) + " result[1]=" + format(result[1]) + " result[2] " + format(
                    result[2]) + "result[3]=" + format(result[3])
                h = result[3] - result[2]
                w = result[1] - result[0]
                print "h=" + format(h)
                print "w=" + format(w)

                if (h < 28):
                    h_1 = ((28 - h) / 2)
                if (w < 28):
                    w_1 = ((28 - w) / 2)

                x1 = result[0] - w_1
                y1 = result[2] - h_1
                x11 = x1.astype('uint8')
                y11 = y1.astype('uint8')
                print "x1=" + format(x1.astype('uint8'))
                print "y1=" + format(y1.astype('uint8'))
                print "Slika 1"
                # plt.imshow(imageDigit2,'gray')
                # plt.show()
                print "Slika 2"

                imageDigit1 = imageDigit2[x11:x11 + 28, y11:y11 + 28]
                digitImages.append(imageDigit1)

            else:
                x1 = result[0] - w_1
                y1 = result[2] - h_1
                x11 = x1.astype('uint8')
                y11 = y1.astype('uint8')
                print "centroide +-15 su izvan dozvoljenog opsega"
                imageDigit = img[y1:y1 + 28, x1:x1 + 28]
                digitImages.append(imageDigit)
                imageBoundary.append((y1, x1))
                validRegions.append(region)
    return digitImages



def dot(v,w):
    x,y = v
    X,Y = w
    return x*X + y*Y
  
def length(v):
    x,y = v
    return math.sqrt(x*x + y*y)
  
def vector(b,e):
    x,y = b
    X,Y = e
    return (X-x, Y-y)
  
def unit(v):
    x,y = v
    mag = length(v)
    return (x/mag, y/mag)
  
def distance(p0,p1):
    return length(vector(p0,p1))
  
def scale(v,sc):
    x,y = v
    return (x * sc, y * sc)
  
def add(v,w):
    x,y = v
    X,Y = w
    return (x+X, y+Y)


def pnt2line2(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
    t = dot(line_unitvec, pnt_vec_scaled)    
    r = 1
    if t < 0.0:
        t = 0.0
        r = -1
    elif t > 1.0:
        t = 1.0
        r = -1
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return (dist, (int(nearest[0]), int(nearest[1])), r)


def putInLeftCorner(img_BW):
    
    minx=700
    miny=700
    maxx=-1
    maxy=-1
    
    newImg="slika"
    
    label_img = label(img_BW)
    regions = regionprops(label_img)
#   
    for region in regions:
        bbox = region.bbox
        if bbox[0]<minx:
            minx=bbox[0]
        if bbox[1] <miny:
            miny=bbox[1]
        if bbox[2]>maxx:
            maxx=bbox[2]
        if bbox[3]>maxy:
            maxy=bbox[3]

    height = maxx - minx
    width = maxy - miny
    
    #read()
    #show()
    

    newImg = np.zeros((28, 28))

    newImg[0:height, 0:width] = newImg[0:height, 0:width] + img_BW[minx:maxx, miny:maxy]

#        plt.show()
    return newImg
def findLabel(digitImg):
    i=0;
    while i<70000:
        sum=0
        mnist_img=new_mnist_set[i]
#        mnist_img=mnist.data[i].reshape(28,28)
      #  new_mnist_img=putInLeftCorner(mnist_img)
        sum=np.sum(mnist_img!=digitImg)
 #       print "suma je " + format(sum)

        if sum<10:

            return mnist.target[i]
        i=i+1


    return -1

def getDigit(img, i):
    #label_img = label(1 - img)
   # img_BW=convertIMG(img)
   # img_gray=(img[:,:,0]+img[:,:,1] + 0* img[:,:,2])/255.0
#    img=[img[:,:,0:1],img[:,:,2]*0]
#    images=findRegions(img)
#    for img in images:
#        plt.imshow(img,'gray')
#        plt.show()
    #=====================VRATITI-START=============================
    img_BW=color.rgb2gray(img) >= 0.88
 #   img_BW=img_gray>=0.88
    img_BW=(img_BW).astype('uint8')
    if(i == 1) :
        str_elem = disk(2)
        img_BW = dilation(img_BW, selem=str_elem)
        str_elem = disk(1)
        img_BW = erosion(img_BW, selem=str_elem)

    #=====================VRATITI-END=============================
#    img_BW=my_rgb2gray(img)==1
#    str_elem=disk(2)
#   img_BW=opening(img_BW,selem=str_elem)
#    img_BW = closing(img_BW, selem=str_elem)
    plt.imshow(img_BW,'gray')
    plt.show()
#===========================putInLeftCorner=============================================

#===========================putInLeftCorner=============================================
    newImg=putInLeftCorner(img_BW)
  #  digit = img_BW[row0:row1, col0:col1]
    #             digit = digit.reshape(784)
 #   img = img0[row0:row1, col0:col1]
    #plt.imshow(newImg, 'gray')
    #plt.show()
   # print("size "+format(newImg.shape))
    rez = findLabel(newImg)
    #              rez=findLabel(digit)
    print("Proslijedjeni broj je " + format(rez))
    return rez


#popunjavanje mnist seta (ucitavanje podataka)
new_mnist_set=[]
def transformMnist(mnist):

    i=0;
    while i < 70000:
        mnist_img=mnist.data[i].reshape(28,28)
        mnist_img_BW=((color.rgb2gray(mnist_img)/255.0)>0.88).astype('uint8')
        #print("Dimenzije mnist image "+format(mnist_img.shape))
#        if(i==2):
#           break
        new_mnist_img=putInLeftCorner(mnist_img_BW)
#        new_mnist_set[i]=new_mnist_img
        new_mnist_set.append(new_mnist_img)
        i=i+1

sve = []


# color filter
def read(dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in xrange(len(lbl)):
        yield get_img(i)

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()


# extract color function - github
def extract_color( src, h_th_low, h_th_up, s_th, v_th ):
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    if h_th_low > h_th_up:
        ret, h_dst_1 = cv2.threshold(h, h_th_low, 255, cv2.THRESH_BINARY) 
        ret, h_dst_2 = cv2.threshold(h, h_th_up,  255, cv2.THRESH_BINARY_INV)

        dst = cv2.bitwise_or(h_dst_1, h_dst_2)
    else:
        ret, dst = cv2.threshold(h,   h_th_low, 255, cv2.THRESH_TOZERO) 
        ret, dst = cv2.threshold(dst, h_th_up,  255, cv2.THRESH_TOZERO_INV)
        ret, dst = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY)

    ret, s_dst = cv2.threshold(s, s_th, 255, cv2.THRESH_BINARY)
    ret, v_dst = cv2.threshold(v, v_th, 255, cv2.THRESH_BINARY)
    dst = cv2.bitwise_and(dst, s_dst)
    dst = cv2.bitwise_and(dst, v_dst)
    return dst

    if __name__ == '__main__':
        param = sys.argv
    if (len(param) != 6):
        print ("Usage: $ python " + param[0] + " sample.jpg h_min, h_max, s_th, v_th")
        quit()  
    # open image file
    try:
        input_img = cv2.imread(param[1])
    except:
        print ('faild to load %s' % param[1])
        quit()

    if input_img is None:
        print ('faild to load %s' % param[1])
        quit()

    # parameter setting
    h_min = int(param[2])
    h_max = int(param[3])
    s_th = int(param[4])
    v_th = int(param[5])
    # making mask using by extract color function
    msk_img = extract_color(input_img, h_min, h_max, s_th, v_th)
    # mask processing
    output_img = cv2.bitwise_and(input_img, input_img, mask = msk_img)

    cv2.imwrite("extract_" + param[1], output_img)


def main():
    transformMnist(mnist)

    for i in range(9, 10, 1):
        
        elements = []
        t =0
        counter = 0
        times = []
        
        suma = 0
        #videoName="Videos/video-0.avi"
        
        #videoName = videoName0 + "-" + format(i) + ".avi"

        cap = cv2.VideoCapture(videoName)

        x1, y1, x2, y2 = findLineParams(videoName)
        # line = [(100, 450), (500, 100)]
        line = [(x1, y1), (x2, y2)]
        
        boundaries = [([230, 230, 230], [255, 255, 255])]
        
        kernel = np.ones((2,2),np.uint8)
        #lower = np.array([230, 230, 230])
        #upper = np.array([255, 255, 255])

        brojac = 0


        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #fourcc = cv2.cv.CV_FOURCC(*'XVID')
        out = cv2.VideoWriter('images/output-rezB.avi',fourcc, 20.0, (640,480))

        while (1):
            start_time = time.time()
            ret, img = cap.read()
            
            brojac = brojac + 1
            
            if not ret:
                break
            (lower, upper) = boundaries[0]
            # create NumPy arrays from the boundaries
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            mask = cv2.inRange(img, lower, upper)
            img0 = 1.0 * mask

            img0 = cv2.dilate(img0, kernel)  # cv2.erode(img0,kernel)
            img0 = cv2.dilate(img0, kernel)
            #pronalazi objekte koji su pronadjeni na slici i jedinstveno ih oznacava i sadrzani su
            #u objektu labeled, a broj pronadjenih objekata se nalazi u promjenljivij nr_objects
            labeled, nr_objects = ndimage.label(img0)
            objects = ndimage.find_objects(labeled)
            
            
            print("Broj objekata je"+ format(brojac))
            
            
            for i in range(nr_objects):
                loc = objects[i]
                (xc, yc) = ((loc[1].stop + loc[1].start) / 2,
                            (loc[0].stop + loc[0].start) / 2)
                (dxc, dyc) = ((loc[1].stop - loc[1].start),
                              (loc[0].stop - loc[0].start))

                if (dxc > 11 or dyc > 11):
                    #VRATI
                   # cv2.circle(img, (xc, yc), 16, (25, 25, 255), 1)
                    elem = {'center': (xc, yc), 'size': (dxc, dyc), 't': t}
                    # find in range
                    #print("centar ({xc},{yc}), size ({dxc},{dyc})".format(xc=xc,yc=yc,dxc=dxc,dyc=dyc))
                    lst = inRange(20, elem, elements)
                    nn = len(lst)
                    if nn == 0:
                        elem['id'] = nextId()
                        elem['t'] = t
                        elem['pass'] = False
                        elem['history'] = [{'center': (xc, yc), 'size': (dxc, dyc), 't': t}]
                        elem['previous'] = img
                        elem['future'] = []
                        elements.append(elem)
                    elif nn == 1:
                        lst[0]['center'] = elem['center']
                        lst[0]['t'] = t
                        lst[0]['history'].append({'center': (xc, yc), 'size': (dxc, dyc), 't': t})
                        lst[0]['previous'] = img
                        lst[0]['future'] = []

            for el in elements:
                tt = t - el['t']
               # x=pnt2line(0,0,0)
                if (tt < 3):
                    dist, pnt, r = pnt2line2(el['center'], line[0], line[1])
                    if r > 0:
                        #vratiti
                      #  cv2.line(img, pnt, el['center'], (0, 255, 25), 1)
                       # c = (25, 25, 255)
                        if (dist < 25
                            ):
                            #c = (0, 255, 160)
                            if el['pass'] == False:
                                el['pass'] = True
                                counter += 1
                                (x,y)=el['center']
                                (sx,sy)=el['size']
                               # getDigit(cv2.circle(img, el['center'], 16, c, 2))
                                x1=x-14
                                x2=x+14
                                y1=y-14
                                y2=y+14
                                #(p1,p2)=(x1,y1)
                                #(p3,p4)=(x2,y2)
                               # cv2.rectangle(img,(p1,p2),(p3,p4),(255,255,0),3)
                                mala = img[y1:y2,x1:x2]
                                rez = getDigit(mala, 0)
                                if(rez>-1):

                                    suma += rez
                                else :
                                    slika = el['previous']
                                    (x,y) = el['history'][len(el['history'])-1]['center']
                                    
                                    #kocka 28x28 oko broja
                                    x1 = x - 14
                                    x2 = x + 14
                                    y1 = y - 14
                                    y2 = y + 14
                                    #(p1, p2) = (x1, y1)
                                    #(p3, p4) = (x2, y2)
                                    # cv2.rectangle(img,(p1,p2),(p3,p4),(255,255,0),3)
                                    prev_mala = slika[y1:y2, x1:x2]
                                    rez = getDigit(prev_mala, 1)
                                    if (rez > -1):

                                        suma += rez
                                print("********************************************************")
                                print ("Pronadjen centar({x},{y}) velicina({sx},{sy})".format(x=x, y=y,sx=sx,sy=sy))
                                print("********************************************************")
                              # print("pnt ".format(el['center']))
                    #vratti promjenljivu umjest c
                   # cv2.circle(img, el['center'], 16, c, 2)
                    #VRATI
                  #  cv2.circle(img, el['center'], 16, (25,25,255), 2)
                   # id = el['id']
                    cv2.putText(img, str(el['id']),
                                (el['center'][0] + 10, el['center'][1] + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
                    for hist in el['history']:
                        ttt = t - hist['t']
                        if (ttt < 100):
                            asd = 1
                            #Zuti trag
                            #cv2.circle(img, hist['center'], 1, (0, 255, 255), 1)

                    for fu in el['future']:
                        ttt = fu[0] - t
                        if (ttt < 100):
                            asd=2
                            # Zuti trag
                            #cv2.circle(img, (fu[1], fu[2]), 1, (255, 255, 0), 1)

            elapsed_time = time.time() - start_time
            times.append(elapsed_time * 1000)
            cv2.putText(img, 'Counter: ' + str(suma), (400, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # print nr_objects
            t += 1
            if t % 10 == 0:
                print t
            cv2.imshow('frame', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            out.write(img)
        out.release()


        et = np.array(times)
        print 'mean %.2f ms' % (np.mean(et))
        sve.append(suma)
    # print np.std(et)
    cap.release()
    cv2.destroyAllWindows()
    print("********************************************************")
    print("********************************************************")
    print ("SUMA SVIH BROJEVA JE:")
    
    print sve
    print("********************************************************")
    print("********************************************************")
main()

