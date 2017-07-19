from PIL import Image
import os, sys

def resizeImage(infile, output_dir="", size=(81,31)):
     outfile = os.path.splitext(infile)[0]+"_resized"
     extension = os.path.splitext(infile)[1]


     if infile != outfile:
        try :
            im = Image.open(infile)
            im.thumbnail(size, Image.ANTIALIAS)
            im.save(output_dir+outfile+extension,extension)
        except IOError:
            print "cannot reduce image for ", infile


if __name__=="__main__":
    output_dir = "uiuc/resized"
    dir = os.getcwd()

    if not os.path.exists(os.path.join(dir,output_dir)):
        os.mkdir(output_dir)

    for file in os.listdir('uiuc/test1'):
        resizeImage(file,output_dir)