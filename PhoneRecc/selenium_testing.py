from time import sleep

from selenium import webdriver

# driver = webdriver.Chrome()
# driver.get('http://127.0.0.1:8000/classifier')

#cd PhoneRecc
#python selenium_testing.py

chromedriver = r"C:\Users\shrik\Downloads\chromedriver_win32\chromedriver.exe"
driver = webdriver.Chrome(executable_path=chromedriver)
driver.get('http://127.0.0.1:8000/classifier')

sleep(5)

driver.find_elements_by_name('battery')[0].send_keys('4000')
driver.find_elements_by_name('int_mem')[0].send_keys('64')
driver.find_elements_by_name('front_camera')[0].send_keys('8')
driver.find_elements_by_name('rear_camera')[0].send_keys('16')
driver.find_elements_by_name('wt')[0].send_keys('200')
driver.find_elements_by_name('cores')[0].send_keys('8')
driver.find_elements_by_name('dual_sim')[0].send_keys('1')
driver.find_elements_by_name('fourg')[0].send_keys('1')
driver.find_elements_by_name('wifi')[0].send_keys('0')
#driver.find_elements_by_name('ram')[0].send_keys('12000')

sleep(2)

driver.find_elements_by_name('submit_btn')[0].click()

sleep(5)

driver.find_elements_by_name('battery')[0].send_keys('HEllo')
driver.find_elements_by_name('int_mem')[0].send_keys('64')
driver.find_elements_by_name('front_camera')[0].send_keys('8')
driver.find_elements_by_name('rear_camera')[0].send_keys('16')
driver.find_elements_by_name('wt')[0].send_keys('200')
driver.find_elements_by_name('cores')[0].send_keys('8')
driver.find_elements_by_name('dual_sim')[0].send_keys('1')
driver.find_elements_by_name('fourg')[0].send_keys('1')
driver.find_elements_by_name('wifi')[0].send_keys('0')
driver.find_elements_by_name('ram')[0].send_keys('12000')

sleep(2)

driver.find_elements_by_name('submit_btn')[0].click()

sleep(5)

driver.find_elements_by_name('battery')[0].send_keys('4000')
driver.find_elements_by_name('int_mem')[0].send_keys('64')
driver.find_elements_by_name('front_camera')[0].send_keys('8')
driver.find_elements_by_name('rear_camera')[0].send_keys('16')
driver.find_elements_by_name('wt')[0].send_keys('200')
driver.find_elements_by_name('cores')[0].send_keys('8')
driver.find_elements_by_name('dual_sim')[0].send_keys('1')
driver.find_elements_by_name('fourg')[0].send_keys('1')
driver.find_elements_by_name('wifi')[0].send_keys('0')
driver.find_elements_by_name('ram')[0].send_keys('12000')

sleep(2)

driver.find_elements_by_name('submit_btn')[0].click()

sleep(5)

driver.close()