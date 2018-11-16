import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

def login(username, password browser=None):
    
    browser.get("https://passport.bilibili.com/login")

    pwd_btn = browser.find_element_by_name("login-username")
    
    act_btn = browser.find_element_by_name("login-password")
    
    submit_btn = browser.find_element_by_name("submit-btn")  

    act_but.send_keys(username)
    
    pwd_btn.send_keys(password)
    
    submint_btn.send_keys(Keys.ENTER)

    return browser

headers = {
        "User-Agent":
            "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36"}

request.headers.update(headers)

cookies = browser.get_cookies()

