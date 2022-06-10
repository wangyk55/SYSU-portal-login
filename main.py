from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait as wait
from selenium.webdriver.support import expected_conditions as EC
import SVM_Train
import cv2
import base64
import numpy as np
from time import sleep
import os

machine = SVM_Train.SVM(1,0.5)
machine.load('D:\WORKSPACE\python\svm.dat')

def IfLoginSuccess():
    login = wait(browser,10).until(EC.visibility_of(browser.find_element_by_xpath('//*[@id="fm1"]/section[2]/input[4]')))
    login.click()
    try:
        browser.find_element_by_xpath('//*[@id="fm1"]/div[1]/span')
        return False
    except:
        return True

def getCaptcha():
    captcha = wait(browser,10).until(EC.visibility_of(browser.find_element_by_xpath('//*[@id="captchaImg"]'))).screenshot_as_base64
    captchaData = base64.b64decode(captcha)
    ImgArray = np.frombuffer(captchaData,np.uint8)
    CaptchaImg = cv2.imdecode(ImgArray,cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(CaptchaImg,cv2.COLOR_BGR2GRAY)
    _,img_bin = cv2.threshold(img_gray,128,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    ch1 = img_bin[:,3:23]
    ch2 = img_bin[:,23:44]
    ch3 = img_bin[:,44:64]
    ch4 = img_bin[:,64:85]
    pct = [ch1,ch2,ch3,ch4]
    pct = SVM_Train.preprocess_hog(pct)
    res = machine.predict(pct)
    veriCode = chr(int(res[0]))
    for i in range(1,4):
        veriCode += chr(int(res[i]))
    return veriCode,[ch1,ch2,ch3,ch4]

options = Options()
options.page_load_strategy = 'none'
browser = webdriver.Chrome(options=options)
try:
    browser.get('https://jwxt.sysu.edu.cn/jwxt/#/login')
    login0 = wait(browser,10).until(EC.visibility_of(browser.find_element_by_xpath('//*[@id="root"]/div/div[2]/div[2]/div[1]/div[3]/div[1]/div[2]/button')))
    login0.click()
except:
    browser.get('https://jwxt-443.webvpn.sysu.edu.cn/jwxt/#/student')
username = 'username'
password = 'password' 
NetID = wait(browser,10).until(EC.visibility_of(browser.find_element_by_xpath('//*[@id="username"]')))
Password = wait(browser,10).until(EC.visibility_of(browser.find_element_by_xpath('//*[@id="password"]')))
Captcha = wait(browser,10).until(EC.visibility_of(browser.find_element_by_xpath('//*[@id="captcha"]')))

veriCode,ch = getCaptcha()

NetID.send_keys(username)
Password.send_keys(password)
Captcha.send_keys(veriCode)
while(not IfLoginSuccess()):
    NetID = wait(browser,10).until(EC.visibility_of(browser.find_element_by_xpath('//*[@id="username"]')))
    Password = wait(browser,10).until(EC.visibility_of(browser.find_element_by_xpath('//*[@id="password"]')))
    Captcha = wait(browser,10).until(EC.visibility_of(browser.find_element_by_xpath('//*[@id="captcha"]')))
    captcha = wait(browser,10).until(EC.visibility_of(browser.find_element_by_xpath('//*[@id="captchaImg"]')))
    veriCode,ch = getCaptcha()
    NetID.send_keys(username)
    Password.send_keys(password)
    Captcha.send_keys(veriCode)
for i in range(0,4):
    num = len([lists for lists in os.listdir('D:\jwxt_login_project/Train/'+veriCode[i]) if os.path.isfile(os.path.join('D:\jwxt_login_project/Train/'+veriCode[i], lists))])
    cv2.imwrite(r'D:\jwxt_login_project/Train/'+veriCode[i]+'/'+str(num)+'.png',ch[i])
SVM_Train.train_svm(r'D:\jwxt_login_project')
try:
    login2 = wait(browser,10).until(EC.visibility_of(browser.find_element_by_xpath('//*[@id="root"]/div/div[2]/div[2]/div[1]/div[3]/div[1]/div[2]/button')))
    login2.click()
finally:
    browser.back()
    sleep(3)
    choose_course = wait(browser,10).until(EC.visibility_of(browser.find_element_by_xpath('//*[@id="root"]/section/div/main/div/div[1]/div[1]/div/div/div[2]/div/div/div[2]/a/div[1]')))
    choose_course.click()
sleep(300)