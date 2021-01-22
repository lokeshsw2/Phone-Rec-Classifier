from django.shortcuts import render
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

#from all_images import all_images_urls

# Create your views here.

#all images url
all_title_images_urls = [['https://m.media-amazon.com/images/I/91eFtaIWpcL._AC_UY218_ML3_.jpg', 'Samsung Galaxy Note 3, Black 32GB (Verizon Wireless)', 0], ['https://m.media-amazon.com/images/I/81ZlbLtZ3PL._AC_UY218_ML3_.jpg', 'Nokia Lumia 900 Black Factory Unlocked', 1], ['https://m.media-amazon.com/images/I/71VMn6229fL._AC_UY218_ML3_.jpg', 'Samsung Galaxy Note 9 Factory Unlocked Phone with 6.4 Screen and 128GB (U.S. Warranty),Ocean Blue', 2], ['https://m.media-amazon.com/images/I/71ap4Pp+y0L._AC_UY218_ML3_.jpg', 'Samsung Galaxy S7 Edge G935FD 32GB Unlocked GSM 4G LTE', 3], ['https://m.media-amazon.com/images/I/814SsOj-45L._AC_UY218_ML3_.jpg', 'Samsung Galaxy Note 5, Black\xa0 64GB (Verizon Wireless)', 4], ['https://m.media-amazon.com/images/I/611pE6W+X0L._AC_UY218_ML3_.jpg', 'Samsung Galaxy Note 5 SM-N920V Gold 32GB (Verizon Wireless)', 5], ['https://m.media-amazon.com/images/I/51L6DbMbvKL._AC_UY218_ML3_.jpg', 'Motorola G6 – 32 GB – Unlocked (AT&T/Sprint/T-Mobile/Verizon) – Deep Indigo - (U.S. Warranty) - PAAE0011US', 6], ['https://m.media-amazon.com/images/I/81VuPb8-arL._AC_UY218_ML3_.jpg', 'Moto G - Verizon Prepaid Phone (Verizon Prepaid Only)', 7], ['https://m.media-amazon.com/images/I/615Y7qMu7lL._AC_UY218_ML3_.jpg', 'Samsung Galaxy Note 8 N950 Factory Unlocked Phone 64GB Midnight Black (Renewed)', 8], ['https://m.media-amazon.com/images/I/51wJeaY3ekL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S8+ 64GB Factory Unlocked Smartphone - 6.2 Screen - US Version (Midnight Black) - US Warranty [SM-G955UZKAXAA]', 9], ['https://m.media-amazon.com/images/I/61WZWpJTYmL._AC_UY218_ML3_.jpg', 'Nokia Lumia 635 8GB Unlocked GSM 4G LTE Windows 8.1 Quad-Core Phone - Black', 10], ['https://m.media-amazon.com/images/I/81suaO+v0mL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S4, White Frost 16GB (AT&T)', 11], ['https://m.media-amazon.com/images/I/61alJun3JvL._AC_UY218_ML3_.jpg', 'Samsung Galaxy J5 SM-J500H/DS GSM Factory Unlocked Smartphone, International Version (Gold)', 12], ['https://m.media-amazon.com/images/I/91s1UjlYJHL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S5 G900V Verizon 4G LTE Smartphone w/ 16MP Camera - Black - Verizon', 13], ['https://m.media-amazon.com/images/I/611v1lZLoJL._AC_UY218_ML3_.jpg', 'Samsung Galaxy A10 32GB (A105M) 6.2 HD+ Infinity-V 4G LTE Factory Unlocked GSM Smartphone - Black', 14], ['https://m.media-amazon.com/images/I/51V-XG0uBDL._AC_UY218_ML3_.jpg', 'Samsung Galaxy A20 US Version Factory Unlocked Cell Phone with 32GB Memory, 6.4 Screen, [SM-A205UZKAXAA], 12 Month Samsung US Warranty,GSM & CDMA Compatible, Black', 15], ['https://m.media-amazon.com/images/I/81a51J6cq9L._AC_UY218_ML3_.jpg', 'Huawei Mate 10 Pro Unlocked Phone, 6 6GB/128GB, AI Processor, Dual Leica Camera, Water Resistant IP67, GSM Only - Mocha Brown (US Warranty)', 16], ['https://m.media-amazon.com/images/I/31MTdGW4xHL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S7 EDGE G935V 32GB, Verizon/GSM Unlocked, (Renewed) (Black)', 17], ['https://m.media-amazon.com/images/I/41K7nb5aDBL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S7 Edge G935V 32GB Gold - Verizon (Renewed)', 18], ['https://m.media-amazon.com/images/I/71s9fDfT7UL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S4, Black Mist 16GB (Sprint)', 19], ['https://m.media-amazon.com/images/I/81VMO0UpxfL._AC_UY218_ML3_.jpg', 'Motorola DROID Turbo XT1254, Black Ballistic Nylon 32GB (Verizon Wireless)', 20], ['https://m.media-amazon.com/images/I/61EzDilvK7L._AC_UY218_ML3_.jpg', 'Samsung Galaxy S7 - Black - 32GB - Verizon (Renewed)', 21], ['https://m.media-amazon.com/images/I/819xBtcnz4L._AC_UY218_ML3_.jpg', 'HUAWEI P20 Lite (32GB + 4GB RAM) 5.84 FHD+ Display, 4G LTE DualSIM GSM Factory Unlocked Smartphone ANE-LX3 - International Model - No Warranty (Sakura Pink)', 22], ['https://m.media-amazon.com/images/I/61291x3og8L._AC_UY218_ML3_.jpg', 'Nokia 6 - 32 GB - Dual Sim Unlocked Smartphone (AT&T/T-Mobile/Metropcs/Cricket/Mint) - Update To Android 9.0 PIE - 5.FHD Screen - Blue - U.S. Warranty', 23], ['https://m.media-amazon.com/images/I/71wjfOWQhsL._AC_UY218_ML3_.jpg', 'Samsung Galaxy Note8 N950U 64GB Unlocked GSM LTE Android Phone w/ Dual 12 Megapixel Camera - Midnight Black', 24], ['https://m.media-amazon.com/images/I/81SMrSvqUWL._AC_UY218_ML3_.jpg', 'Samsung Galaxy A50 A505G 64GB Duos GSM Unlocked Phone w/Triple 25MP Camera - Blue', 25], ['https://m.media-amazon.com/images/I/814WwhaRVuL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S6 Edge, White Pearl 32GB (Verizon Wireless)', 26], ['https://m.media-amazon.com/images/I/61ZaskM5hBL._AC_UY218_ML3_.jpg', 'Xiaomi Redmi Note 7, 64GB/4GB RAM, 6.30 FHD+, Snapdragon 660, Black - Unlocked Global Version', 27], ['https://m.media-amazon.com/images/I/71-afm8RuLL._AC_UY218_ML3_.jpg', 'Motorola Droid RAZR M XT907 Verizon Wireless, 8GB, White', 28], ['https://m.media-amazon.com/images/I/61UTfUPGH1L._AC_UY218_ML3_.jpg', 'Nokia Lumia 920, Black 32GB (AT&T)', 29], ['https://m.media-amazon.com/images/I/61MF7kZkrIL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S8 Plus (S8+) (SM-G955FD)4GB RAM / 64GB ROM 6.2-Inch 12MP 4G LTE Dual SIM FACTORY UNLOCKED - International Stock No Warranty (MIDNIGHT BLACK)', 30], ['https://m.media-amazon.com/images/I/61uRnzVNj9L._AC_UY218_ML3_.jpg', 'Nokia 7.1 - Android 9.0 Pie - 64 GB - Dual Camera - Dual SIM Unlocked Smartphone (Verizon/AT&T/T-Mobile/MetroPCS/Cricket/H2O) - 5.84 FHD+ HDR Screen - Steel - U.S. Warranty', 31], ['https://m.media-amazon.com/images/I/51hvE4YI96L._AC_UY218_ML3_.jpg', 'Samsung Galaxy Note 8 N950U 64GB Unlocked GSM 4G LTE Android Smartphone w/Dual 12 MegaPixel Camera (Renewed) (Midnight Black)', 32], ['https://m.media-amazon.com/images/I/71iQ4QgMchL._AC_UY218_ML3_.jpg', 'Nokia 3310 3G - Unlocked Single SIM Feature Phone (AT&T/T-Mobile/MetroPCS/Cricket/Mint) - 2.4 Inch Screen - Charcoal', 33], ['https://m.media-amazon.com/images/I/81A-Ww8FqNL._AC_UY218_ML3_.jpg', 'Sony Xperia XA Ultra (F3213) 4G LTE Unlocked GSM Phone w/ 6 Borderless Display, 21.5MP+16MP Cameras, Octa-Core CPU - White', 34], ['https://m.media-amazon.com/images/I/81HfPVitlfL._AC_UY218_ML3_.jpg', 'Motorola Moto X (2nd generation) XT1097 GSM Unlocked Cellphone, 16GB, Black Soft Touch', 35], ['https://m.media-amazon.com/images/I/61EfvRF3sHL._AC_UY218_ML3_.jpg', 'Google Pixel XL Phone 32GB - 5.5 inch display ( Factory Unlocked US Version ) (Very Silver)', 36], ['https://m.media-amazon.com/images/I/91Q7P86ef5L._AC_UY218_ML3_.jpg', 'Samsung Galaxy Note 3 N900A 32GB Unlocked GSM Octa-Core Smartphone w/ 13MP Camera - Black', 37], ['https://m.media-amazon.com/images/I/41cIrncoPlL._AC_UY218_ML3_.jpg', 'Samsung Galaxy A70 128GB/6GB SM-A705M/DS 6.7 HD+ Infinity-U 4G/LTE Factory Unlocked Smartphone (International Version , No Warranty) (Blue)', 38], ['https://m.media-amazon.com/images/I/61w4AKhyLzL._AC_UY218_ML3_.jpg', 'Unlocked GOLD Xiaomi Mi A2, 4GB 64GB, Dual SIM standby, Global Version, 5.5 inch Smartphone Android One, Dual Rear 12.0MP Camera Snapdragon 625', 39], ['https://m.media-amazon.com/images/I/517Q3-wHBkL._AC_UY218_ML3_.jpg', 'Xiaomi Redmi Note 6 Pro 64GB / 4GB RAM 6.26 Dual Camera LTE Factory Unlocked Smartphone Global Version (Black)', 40], ['https://m.media-amazon.com/images/I/71PZz7CQ9UL._AC_UY218_ML3_.jpg', 'Google Pixel XL G2PW210032GBBK Factory Unlocked Smartphone, 32GB, 5.5-Inch Display - U.S. Version (Quite Black)', 41], ['https://m.media-amazon.com/images/I/71nplhIIYjL._AC_UY218_ML3_.jpg', 'Moto G7 Power - Unlocked - 64 GB - Marine Blue (No Warranty) - International Model (GSM Only)', 42], ['https://m.media-amazon.com/images/I/41FBnbqW3pL._AC_UY218_ML3_.jpg', 'Xiaomi Redmi Note 7, 64GB/4GB RAM, 6.30 FHD+, Snapdragon 660, Blue - Unlocked Global Version, No Warranty', 43], ['https://m.media-amazon.com/images/I/419J7KwTxML._AC_UY218_ML3_.jpg', 'Samsung Galaxy S7 SM-G930A AT&T Unlocked Smartphone, (Black Onyx)', 44], ['https://m.media-amazon.com/images/I/51eM8j7wKzL._AC_UY218_ML3_.jpg', 'Huawei Mate 20 Lite SNE-LX3 64GB (Factory Unlocked) 6.3 FHD (International Version) (Black)', 45], ['https://m.media-amazon.com/images/I/81ba7gOGt5L._AC_UY218_ML3_.jpg', 'Nokia Lumia 530 White -No Contract (T-Mobile)', 46], ['https://m.media-amazon.com/images/I/5103hv5OP0L._AC_UY218_ML3_.jpg', 'Samsung Galaxy J7 - Verizon Carrier Locked No Contract Prepaid Smartphone', 47], ['https://m.media-amazon.com/images/I/51cRE43zKwL._AC_UY218_ML3_.jpg', 'Apple iPhone 7 32GB, Rose Gold (Renewed)', 48], ['https://m.media-amazon.com/images/I/61+mrwyL24L._AC_UY218_ML3_.jpg', 'Apple iPhone 6S, 64GB, Rose Gold - For AT&T / T-Mobile (Renewed)', 49], ['https://m.media-amazon.com/images/I/81yZOQEC+NL._AC_UY218_ML3_.jpg', 'Apple iPhone X, 256GB, Silver - For AT&T / T-Mobile (Renewed)', 50], ['https://m.media-amazon.com/images/I/81yZOQEC+NL._AC_UY218_ML3_.jpg', 'Apple iPhone X, 256GB, Silver - For AT&T (Renewed)', 51], ['https://m.media-amazon.com/images/I/81hxAMIxAeL._AC_UY218_ML3_.jpg', 'Samsung Galaxy Rugby Pro 4G LTE I547 Unlocked Android Ruggedized Smart Phone', 52], ['https://m.media-amazon.com/images/I/719knfTwPvL._AC_UY218_ML3_.jpg', 'Apple iPhone X, Unlocked 5.8, 64GB - Space Gray (Renewed)', 53], ['https://m.media-amazon.com/images/I/81vLTvMW6DL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S4, White Frost 16GB (Sprint)', 54], ['https://m.media-amazon.com/images/I/81s0xgVWDOL._AC_UY218_ML3_.jpg', 'Samsung Focus I917 Unlocked Phone with Windows 7 OS, 5 MP Camera, and Wi-Fi--No Warranty (Black)', 55], ['https://m.media-amazon.com/images/I/41zkjf9ksSL._AC_UY218_ML3_.jpg', 'Nokia Lumia 928, White 32GB (Verizon Wireless)', 56], ['https://m.media-amazon.com/images/I/81k6Nq0KI1L._AC_UY218_ML3_.jpg', 'Motorola Barrage V860 Phone (Verizon Wireless)', 57], ['https://m.media-amazon.com/images/I/41oBClPPoCL._AC_UY218_ML3_.jpg', 'Apple iPhone 6S, 16GB, Rose Gold - For AT&T / T-Mobile (Renewed)', 58], ['https://m.media-amazon.com/images/I/61-rum+PvIL._AC_UY218_ML3_.jpg', 'Nokia 2 - Android - 8GB - Dual SIM Unlocked Smartphone (AT&T/T-Mobile/MetroPCS/Cricket/H2O) - 5 Screen - Black - U.S. Warranty', 59], ['https://m.media-amazon.com/images/I/71Q7aSSyOkL._AC_UY218_ML3_.jpg', 'Samsung Convoy 3, Gray (Verizon Wireless)', 60], ['https://m.media-amazon.com/images/I/81oYNE1MUxL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S6 G920A 32GB Unlocked GSM 4G LTE Octa-Core Android Smartphone with 16MP Camera - Black Sapphire', 61], ['https://m.media-amazon.com/images/I/7108ZFza5gL._AC_UY218_ML3_.jpg', 'Google Pixel 2 64 GB, Black Factory Unlocked (Renewed)', 62], ['https://m.media-amazon.com/images/I/71UTxbIjXaL._AC_UY218_ML3_.jpg', 'Motorola XT1775 Moto E Plus (4th Gen.) 32GB Unlocked Fine Gold Smartphone', 63], ['https://m.media-amazon.com/images/I/81s7ZLOGOWL._AC_UY218_ML3_.jpg', 'Apple iPhone 6S, 64GB, Space Gray - Fully Unlocked (Renewed)', 64], ['https://m.media-amazon.com/images/I/61NDR6WOqPL._AC_UY218_ML3_.jpg', 'Samsung Galaxy Prevail II (Boost Mobile) (Discontinued by Manufacturer)', 65], ['https://m.media-amazon.com/images/I/81zlazvfjBL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S6 Active, 32 GB , Grey (AT&T)', 66], ['https://m.media-amazon.com/images/I/71Y+FpZYcFL._AC_UY218_ML3_.jpg', 'Samsung a157 GoPhone (AT&T)', 67], ['https://m.media-amazon.com/images/I/71SWW5LsZ0L._AC_UY218_ML3_.jpg', 'Samsung Galaxy S9 Plus Verizon + GSM Unlocked 64GB Midnight Black (Renewed)', 68], ['https://m.media-amazon.com/images/I/817MhreagcL._AC_UY218_ML3_.jpg', 'Sony Xperia XA1 Ultra G3223 32GB Unlocked GSM LTE Octa-Core Phone w/ 23MP - Gold', 69], ['https://m.media-amazon.com/images/I/81o50fc16kL._AC_UY218_ML3_.jpg', 'Google Pixel GSM Unlocked (Renewed) (32GB, Gray)', 70], ['https://m.media-amazon.com/images/I/71se2LK4Y5L._AC_UY218_ML3_.jpg', 'Moto Z GSM Unlocked Smartphone,5.5 Quad HD screen, 64GB storage, 5.2mm thin - Black', 71], ['https://m.media-amazon.com/images/I/81Vobb06FVL._AC_UY218_ML3_.jpg', 'Moto G7 – Unlocked – 64 GB – Ceramic Black (US Warranty) - Verizon, AT&T, T-Mobile, Sprint, Boost, Cricket, & Metro', 72], ['https://m.media-amazon.com/images/I/71z1TjwnadL._AC_UY218_ML3_.jpg', 'Google Pixel Phone - 5 inch display (Factory Unlocked US Version) (32GB, Quite Black)', 73], ['https://m.media-amazon.com/images/I/415nK4G4hJL._AC_UY218_ML3_.jpg', 'Samsung a157V (AT&T Go Phone) No Annual Contract', 74], ['https://m.media-amazon.com/images/I/719hX34RZhL._AC_UY218_ML3_.jpg', 'Sony Xperia XA2 Ultra Factory Unlocked Phone - 6 Screen- 32GB - Silver (U.S. Warranty)', 75], ['https://m.media-amazon.com/images/I/61uo-nK+OAL._AC_UY218_ML3_.jpg', 'Motorola XT1585 Turbo 2, (32GB) 5.4 (White, Verizon)', 76], ['https://m.media-amazon.com/images/I/61MAsozHcaL._AC_UY218_ML3_.jpg', 'Samsung Galaxy Note 5 SM-N920V 32GB White Smartphone for Verizon (Renewed)', 77], ['https://m.media-amazon.com/images/I/81ZGi56SecL._AC_UY218_ML3_.jpg', 'Samsung Galaxy Note 5 32GB GSM Unlocked - Black (Renewed) (D132)', 78], ['https://m.media-amazon.com/images/I/711v+hjDjxL._AC_UY218_ML3_.jpg', 'Samsung Galaxy A10 A105M 32GB Duos GSM Unlocked Phone w/ 13MP Camera - Blue', 79], ['https://m.media-amazon.com/images/I/61TQfjuS1EL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S7 Edge 32GB G935A GSM Unlocked (Renewed) (Black)', 80], ['https://m.media-amazon.com/images/I/71i9lcnWT1L._AC_UY218_ML3_.jpg', 'Samsung Galaxy A30 Dual SIM 32GB (SM-A305G/DS) Unlocked Phone GSM International Version - Blue', 81], ['https://m.media-amazon.com/images/I/51H2fS7s9FL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S7 Edge G935A 32GB Gold - Unlocked GSM (Renewed)', 82], ['https://m.media-amazon.com/images/I/51ZL33qmCpL._AC_UY218_ML3_.jpg', 'Samsung Gusto 3, Royal Navy Blue (Verizon Wireless Prepaid) - Discontinued by Manufacturer', 83], ['https://m.media-amazon.com/images/I/717w0Z516KL._AC_UY218_ML3_.jpg', 'Motorola DROID MAXX, Black 16GB (Verizon Wireless)', 84], ['https://m.media-amazon.com/images/I/71Q7aSSyOkL._AC_UY218_ML3_.jpg', 'Samsung Convoy 3 SCH-U680 Rugged 3G Cell Phone Verizon Wireless', 85], ['https://m.media-amazon.com/images/I/61Y6BSxzezL._AC_UY218_ML3_.jpg', 'Samsung Galaxy Note 10+ Plus Factory Unlocked Cell Phone with 512GB (U.S. Warranty), Aura Black/ Note10+', 86], ['https://m.media-amazon.com/images/I/61YVqHdFRxL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S10 128GB+8GB RAM SM-G973F/DS Dual Sim 6.1LTE Factory Unlocked Smartphone (International Model No Warranty) (Prism White)', 87], ['https://m.media-amazon.com/images/I/71CDE9pG4hL._AC_UY218_ML3_.jpg', 'Google Pixel 2 XL 128 GB, Black (Renewed)', 88], ['https://m.media-amazon.com/images/I/61XyeFgc3vL._AC_UY218_ML3_.jpg', 'Sony Xperia XZ F8332 64GB Forest Blue, 5.2, Dual Sim, GSM Unlocked International Model, No Warranty', 89], ['https://m.media-amazon.com/images/I/91xMuzi75uL._AC_UY218_ML3_.jpg', 'Verizon Samsung Convoy U660 No Contract Rugged PTT Cell Phone Grey Verizon', 90], ['https://m.media-amazon.com/images/I/71Totr78FmL._AC_UY218_ML3_.jpg', 'Sony Xperia XZ F8331 32GB Unlocked GSM 4G LTE Phone w/ 23MP Camera - Mineral Black', 91], ['https://m.media-amazon.com/images/I/51x8eZ8JbKL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S10 Factory Unlocked Phone with 128GB - Prism Black', 92], ['https://m.media-amazon.com/images/I/51i3dF4frhL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S7 32GB G930V Unlocked - Black', 93], ['https://m.media-amazon.com/images/I/71kSYOju0xL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S7 32GB Unlocked (Verizon Wireless) - Gold', 94], ['https://m.media-amazon.com/images/I/81Z0DX6B3bL._AC_UY218_ML3_.jpg', 'Samsung GALAXY S6 G920 32GB Unlocked GSM 4G LTE Octa-Core Smartphone - Black Sapphire', 95], ['https://m.media-amazon.com/images/I/71wchmqQn+L._AC_UY218_ML3_.jpg', 'Sony Xperia XA1 G3123 32GB Unlocked GSM LTE Octa-Core Phone w/ 23MP Camera - White', 96], ['https://m.media-amazon.com/images/I/71CDE9pG4hL._AC_UY218_ML3_.jpg', 'Google Pixel 2 XL 64 GB, Black (Renewed)', 97], ['https://m.media-amazon.com/images/I/51X1YcLSmXL._AC_UY218_ML3_.jpg', 'Xiaomi Redmi Note 7 128GB + 4GB RAM 6.3 FHD+ LTE Factory Unlocked 48MP GSM Smartphone (Global Version, No Warranty) (Neptune Blue)', 98], ['https://m.media-amazon.com/images/I/A1S4AmmN0pL._AC_UY218_ML3_.jpg', 'Samsung Galaxy Note 4, Frosted White 32GB (Sprint)', 99], ['https://m.media-amazon.com/images/I/71q3UGwZhcL._AC_UY218_ML3_.jpg', 'Xiaomi Pocophone F1 128GB Graphite Black, Dual Sim, 6GB RAM, Dual Camera, 6.18, GSM Unlocked Global Model, No Warranty', 100], ['https://m.media-amazon.com/images/I/61oKJ6RYCDL._AC_UY218_ML3_.jpg', 'Xiaomi Mi 8 Lite(64GB, 4GB RAM) 6.26 Full Screen Display, Snapdragon 660, Dual AI Cameras, Factory Unlocked Phone - International Global 4G LTE Version (Black)', 101], ['https://m.media-amazon.com/images/I/71KB5PiwJAL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S7, Black 32GB (Verizon Wireless)', 102], ['https://m.media-amazon.com/images/I/61pVtPaTkML._AC_UY218_ML3_.jpg', 'Verizon Wireless Motorola RAZR V3m - Silver', 103], ['https://m.media-amazon.com/images/I/61y-TkZ5lWL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S7 Edge G935A 32GB AT&T - Black Onyx', 104], ['https://m.media-amazon.com/images/I/51zGEG7p0FL._AC_UY218_ML3_.jpg', 'Samsung Galaxy J2 Core 2018 International Version, No Warranty Factory Unlocked 4G LTE (USA Latin Caribbean) Android Oreo SM-J260M Dual Sim 8MP 8GB (Black)', 105], ['https://m.media-amazon.com/images/I/91oY6n78hhL._AC_UY218_ML3_.jpg', 'Motorola Droid Turbo - 32GB Android Smartphone - Verizon - Black (Renewed)', 106], ['https://m.media-amazon.com/images/I/61+387TW4-L._AC_UY218_ML3_.jpg', 'Sony H3123 - Black/SBH-90C/SCSH10 Xperia XA2 Accessory Bundle, (Bundle Includes: 1 Xperia XA2, 1 SBH90C Bluetooth Headset, 1 Flip Phone Case), Black', 107], ['https://m.media-amazon.com/images/I/81ZwjKulg8L._AC_UY218_ML3_.jpg', 'Sony Xperia X F5121 32GB GSM 23MP Camera Phone - Graphite Black', 108], ['https://m.media-amazon.com/images/I/814kh7KdbtL._AC_UY218_ML3_.jpg', 'Telcel America Motorola Pre-Paid Cell Phone - Motogo! EX431G - Black', 109], ['https://m.media-amazon.com/images/I/71kLFOLKN3L._AC_UY218_ML3_.jpg', 'Samsung Galaxy A50 US Version Factory Unlocked Cell Phone with 64GB Memory, 6.4 Screen, Black, [SM-A505UZKNXAA]', 110], ['https://m.media-amazon.com/images/I/51C6HtwGL+L._AC_UY218_ML3_.jpg', 'Samsung Galaxy S7 G930T 32GB T-Mobile - Black', 111], ['https://m.media-amazon.com/images/I/81CgLTDOqQL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S7 32GB T-Mobile - Gold Platinum', 112], ['https://m.media-amazon.com/images/I/51Hc4HmwwbL._AC_UY218_ML3_.jpg', 'Xiaomi Redmi Note 7 128GB + 4GB RAM 6.3 FHD+ LTE Factory Unlocked 48MP GSM Smartphone (Global Version, No Warranty) (Space Black)', 113], ['https://m.media-amazon.com/images/I/71RYhD1uzpL._AC_UY218_ML3_.jpg', 'Apple iPhone 7, 128GB, Gold - For AT&T / T-Mobile (Renewed)', 114], ['https://m.media-amazon.com/images/I/61gnQwobQHL._AC_UY218_ML3_.jpg', 'Apple iPhone 6S Plus, 16GB, Silver - For AT&T / T-Mobile (Renewed)', 115], ['https://m.media-amazon.com/images/I/81Fr+G5BcfL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S6 G920 32GB Unlocked GSM 4G LTE Octa-Core Smartphone, Gold Platinum', 116], ['https://m.media-amazon.com/images/I/71x3e0x+M2L._AC_UY218_ML3_.jpg', 'Apple iPhone 7, 32GB, Rose Gold - For AT&T / T-Mobile (Renewed)', 117], ['https://m.media-amazon.com/images/I/7159JihtggL._AC_UY218_ML3_.jpg', 'Motorola Moto X4 Factory Unlocked Phone - 32GB - 5.2 Super Black - PA8S0006US', 118], ['https://m.media-amazon.com/images/I/61C9GrXEp4L._AC_UY218_ML3_.jpg', 'Google Pixel 4 - Just Black - 64GB - Unlocked', 119], ['https://m.media-amazon.com/images/I/41CU2Axt3fL._AC_UY218_ML3_.jpg', 'Google Pixel 3 XL 64GB Unlocked GSM & CDMA 4G LTE Android Phone w/ 12.2MP Rear & Dual 8MP Front Camera - Just Black (Renewed)', 120], ['https://m.media-amazon.com/images/I/81suaO+v0mL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S4 SGH-I337 USA GSM Unlocked Cellphone, 16GB, Frost White', 121], ['https://m.media-amazon.com/images/I/71-e3enEyYL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S7 Edge, 5.5 32GB (Verizon Wireless) - Blue', 122], ['https://m.media-amazon.com/images/I/511i4Qx+e2L._AC_UY218_ML3_.jpg', 'Motorola Moto G6 Plus - 64GB - 5.9 FHD+, Dual SIM 4G LTE GSM Factory Unlocked Smartphone International Model XT1926-7 (Deep Indigo)', 123], ['https://m.media-amazon.com/images/I/41RWkB9NDEL._AC_UY218_ML3_.jpg', 'Samsung Galaxy J3 Prime J327A 16GB 4G LTE 7.0 Nougat 5 GSM Unlocked - Black', 124], ['https://m.media-amazon.com/images/I/51BateZ5iqL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S10e 128GB+6GB RAM SM-G970 Dual Sim 5.8 LTE Factory Unlocked Smartphone (International Model) (Prism Black)', 125], ['https://m.media-amazon.com/images/I/71RO-IMRSvL._AC_UY218_ML3_.jpg', 'Motorola Moto G6 (32GB, 3GB RAM) Dual SIM 5.7 4G LTE (GSM Only) Factory Unlocked Smartphone International Model XT1925-2 (Deep Indigo)', 126], ['https://m.media-amazon.com/images/I/81nfrNby6HL._AC_UY218_ML3_.jpg', 'Sony Xperia XZ Premium - Unlocked Smartphone - 5.5, 64GB - Dual SIM - Pink (US Warranty)', 127], ['https://m.media-amazon.com/images/I/51Xm9ay971L._AC_UY218_ML3_.jpg', 'Xiaomi Mi A3 64GB + 4GB RAM, Triple Camera, 4G LTE Smartphone - International Global Version (Not just Blue)', 128], ['https://m.media-amazon.com/images/I/41kBtX8-WBL._AC_UY218_ML3_.jpg', 'Xiaomi Mi A3 64GB, 4GB RAM 6.1 48MP AI Triple Camera LTE Factory Unlocked Smartphone (International Version) (Kind of Grey)', 129], ['https://m.media-amazon.com/images/I/61Mc+wla27L._AC_UY218_ML3_.jpg', 'Samsung Galaxy J1 Mini 8GB J106H/DS Dual Sim Unlocked Phone - Retail Packaging- Black', 130], ['https://m.media-amazon.com/images/I/71XeQzRDyML._AC_UY218_ML3_.jpg', 'Apple iPhone Xs Max, 256GB, Space Gray - Fully Unlocked (Renewed)', 131], ['https://m.media-amazon.com/images/I/81nSsCFeiTL._AC_UY218_ML3_.jpg', 'Apple iPhone XS Max, 256GB, Gray - For AT&T (Renewed)', 132], ['https://m.media-amazon.com/images/I/51V8qIvvd1L._AC_UY218_ML3_.jpg', 'Samsung Galaxy S10+ Plus 128GB+8GB RAM SM-G975F/DS Dual Sim 6.4 LTE Factory Unlocked Smartphone International Model, No Warranty (Prism Black)', 133], ['https://m.media-amazon.com/images/I/51W3p2zTv9L._AC_UY218_ML3_.jpg', 'Nokia Lumia Icon, Black 32GB (Verizon Wireless)', 134], ['https://m.media-amazon.com/images/I/51Hy0ypovHL._AC_UY218_ML3_.jpg', 'Motorola Moto One - Android One - 64 GB - 13+2 MP Dual Rear Camera - Dual SIM Unlocked Smartphone (at&T/T-Mobile/MetroPCS/Cricket/H2O) - 5.9 HD+ Display - XT1941-3 - (International Version) (Black)', 135], ['https://m.media-amazon.com/images/I/81h0cGu5OcL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S6 Edge+, Black 64GB (Sprint)', 136], ['https://m.media-amazon.com/images/I/61wgvFSAJQL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S8Active 64GB SM-G892A Unlocked GSM - Meteor Gray (Renewed)', 137], ['https://m.media-amazon.com/images/I/71c5RZEpLwL._AC_UY218_ML3_.jpg', 'Apple iPhone 7 Plus 256GB Unlocked GSM 4G LTE Quad-Core Smartphone - Jet Black (Renewed)', 138], ['https://m.media-amazon.com/images/I/41YWHpWk7+L._AC_UY218_ML3_.jpg', 'Motorola MOTO Z PLAY (XT1635) Factory Unlocked Phone - 5.5 Screen - 32GB - Black (International Version - No Warranty)', 139], ['https://m.media-amazon.com/images/I/71lpZjjF7NL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S6 (SM-G920V) - 32GB Verizon + GSM Smartphone - Black Sapphire (Renewed)', 140], ['https://m.media-amazon.com/images/I/41gYAzWwF1L._AC_UY218_ML3_.jpg', 'Samsung Galaxy S7 Active G891A 32GB Unlocked GSM Shatter,Dust and Water Resistant Smartphone w/ 12MP Camera (AT&T) - Sandy Gold', 141], ['https://m.media-amazon.com/images/I/71QHUuh-ctL._AC_UY218_ML3_.jpg', 'ASUS ZenFone Max Plus ZB570TL-MT67-3G32G-BL - 5.7” 1920x1080-3GB RAM - 32GB storage - LTE Unlocked Dual SIM Cell Phone - US Warranty - Silver', 142], ['https://m.media-amazon.com/images/I/71RCiZl-V0L._AC_UY218_ML3_.jpg', 'Samsung Galaxy S7 G930V 32GB, Verizon, Black Onyx, Unlocked Smartphones (Renewed)', 143], ['https://m.media-amazon.com/images/I/81KgaU7qznL._AC_UY218_ML3_.jpg', 'Google Pixel 2 128GB Unlocked GSM/CDMA 4G LTE Octa-Core Phone w/ 12.2MP Camera - Just Black', 144], ['https://m.media-amazon.com/images/I/51p8LMQ-rIL._AC_UY218_ML3_.jpg', 'Samsung R355C Net 10 Unlimited', 145], ['https://m.media-amazon.com/images/I/715rN0Y8PqL._AC_UY218_ML3_.jpg', 'Google Pixel 2 GSM/CDMA Google Unlocked (Clearly White, 64GB, US warranty)', 146], ['https://m.media-amazon.com/images/I/81dXcgzgqkL._AC_UY218_ML3_.jpg', 'Google Pixel 2 XL 64GB Unlocked GSM/CDMA 4G LTE Octa-Core Phone w/ 12.2MP Camera - Just Black', 147], ['https://m.media-amazon.com/images/I/81cvwVzNoiL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S6 G920T 32GB Unlocked GSM 4G LTE Octa-Core Android Smartphone w/ 16MP Camera - Black', 148], ['https://m.media-amazon.com/images/I/71Bpvs8eUWL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S4 16GB Black SPH-L720T Tri-Band (Boost Mobile)', 149],['https://m.media-amazon.com/images/I/61lfpjOekDL._AC_UY218_ML3_.jpg', 'Xiaomi Mi A2 Lite (64GB, 4GB RAM) 5.84 18:9 HD Display, Dual Camera, Android One Unlocked Smartphone - International Global LTE Version (Gold)', 150], ['https://m.media-amazon.com/images/I/614sFnZbspL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S6 SM-G920A 32GB White Smartphone for AT&T (Renewed)', 151], ['https://m.media-amazon.com/images/I/61Uy8S7wD9L._AC_UY218_ML3_.jpg', 'Samsung Galaxy S4 16GB SPH-L720 4G LTE Android - Sprint (Blue)', 152], ['https://m.media-amazon.com/images/I/41+2tWGDs3L._AC_UY218_ML3_.jpg', 'Apple iPhone XS, 256GB, Gold - Fully Unlocked (Renewed)', 153], ['https://m.media-amazon.com/images/I/61xa10dafvL._AC_UY218_ML3_.jpg', 'Apple iPhone XS, 256GB, Gray - For AT&T (Renewed)', 154], ['https://m.media-amazon.com/images/I/61+mDJSfYuL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S10e Factory Unlocked Phone with 128GB, Prism White (Renewed)', 155], ['https://m.media-amazon.com/images/I/810MbmOEoqL._AC_UY218_ML3_.jpg', 'Apple iPhone 8 Plus 64GB Unlocked GSM Phone - Space Gray (Renewed)', 156], ['https://m.media-amazon.com/images/I/61IR7+oiS7L._AC_UY218_ML3_.jpg', 'Samsung Galaxy S10e Factory Unlocked Phone with 128GB (U.S. Warranty), Prism Blue (Renewed)', 157], ['https://m.media-amazon.com/images/I/91JHyj8K0FL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S6 32GB G920A AT&T Unlocked - Gold Platinum', 158], ['https://m.media-amazon.com/images/I/71ofnXiUFbL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S7 G930P 32GB Gold - Sprint (Renewed)', 159], ['https://m.media-amazon.com/images/I/41QNnygel4L._AC_UY218_ML3_.jpg', 'Nokia 105 [2017] TA-1037 Only 2G Dual-Band (850/1900) Factory Unlocked Mobile Phone Black no warranty (White)', 160], ['https://m.media-amazon.com/images/I/61QcWsvnpQL._AC_UY218_ML3_.jpg', 'Huawei Mate 20 Pro LYA-L29 128GB + 6GB - Factory Unlocked International Version - GSM ONLY, NO CDMA - No Warranty in The USA (Black)', 161], ['https://m.media-amazon.com/images/I/71b4kAq+5QL._AC_UY218_ML3_.jpg', 'Apple iPhone 7 Plus 256GB Unlocked GSM Quad-Core Phone - Black (Renewed)', 162], ['https://m.media-amazon.com/images/I/81UpiOZp47L._AC_UY218_ML3_.jpg', 'Motorola Moto X - 2nd Generation, Black Resin 16GB (Verizon Wireless)', 163], ['https://m.media-amazon.com/images/I/81O5HRPD1gL._AC_UY218_ML3_.jpg', 'Samsung Focus Flash I677 8GB Unlocked GSM Phone with Windows 7.5 OS, 5MP Camera, GPS, Wi-Fi, Bluetooth and FM Radio - Black', 164], ['https://m.media-amazon.com/images/I/51X4GhYfx4L._AC_UY218_ML3_.jpg', 'Nokia 105 RM-1135 Dual-Band (850/1900 MHz) Factory Unlocked Mobile Phone, Black, 2G Network Only.', 165], ['https://m.media-amazon.com/images/I/61ktS3pR0TL._AC_UY218_ML3_.jpg', 'Motorola Moto G4 Play (4th Generation) 16GB 4G LTE Unlocked ONLY GSM 5 Inches International Version No Warranty (White)', 166], ['https://m.media-amazon.com/images/I/61vD0TwZjdL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S6, G920P White Pearl 32GB - Sprint (Renewed)', 167], ['https://m.media-amazon.com/images/I/81qwFH3PTCL._AC_UY218_ML3_.jpg', 'ROG Phone Gaming Smartphone ZS600KL-S845-8G512G - 6 FHD+ 2160x1080 90Hz Display - Qualcomm Snapdragon 845 - 8GB RAM - 512GB Storage - LTE Unlocked Dual SIM Gaming Phone - US Warranty', 168], ['https://m.media-amazon.com/images/I/71pjGN8WsoL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S8 64GB Phone -5.8in Unlocked Smartphone - Midnight Black (Renewed)', 169], ['https://m.media-amazon.com/images/I/71rzaPrNXrL._AC_UY218_ML3_.jpg', 'Sony Xperia X Performance F8131 32GB Unlocked GSM LTE Android Phone w/ 23MP Camera - Black', 170], ['https://m.media-amazon.com/images/I/51TaayMzqtL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S9 (SM-G960F/DS) 4GB/ 64GB 5.8-inches LTE Dual SIM (GSM Only, No CDMA) Factory Unlocked - International Stock No Warranty (Midnight Black, Phone Only)', 171], ['https://m.media-amazon.com/images/I/81FWIR3RbUL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S8 SM-G950F Unlocked 64GB - International Version/No Warranty (GSM Only, No CDMA) (Midnight Black)', 172], ['https://m.media-amazon.com/images/I/81IpIT71b7L._AC_UY218_ML3_.jpg', 'Samsung Galaxy Alpha, Frosted Gold 32GB (AT&T)', 173], ['https://m.media-amazon.com/images/I/81CFhqN240L._AC_UY218_ML3_.jpg', 'Nokia Lumia 928 32GB Unlocked GSM 4G LTE Windows Smartphone w/ 8MP Carl Zeiss Optics Camera - Black', 174], ['https://m.media-amazon.com/images/I/71cLpzYW9IL._AC_UY218_ML3_.jpg', 'Sony Xperia L1 G3313 16GB Unlocked GSM Quad-Core Android Phone - Pink', 175], ['https://m.media-amazon.com/images/I/81yZOQEC+NL._AC_UY218_ML3_.jpg', 'Apple iPhone X, GSM Unlocked, 256GB - Silver (Renewed)', 176], ['https://m.media-amazon.com/images/I/51ucn49vPUL._AC_UY218_ML3_.jpg', 'Motorola Moto G6 Play 32GB- Dual SIM 5.7 4G LTE (GSM Only) Factory Unlocked Smartphone International Version XT1922-5 (Deep Indigo)', 177], ['https://m.media-amazon.com/images/I/51hnjqGbVqL._AC_UY218_ML3_.jpg', 'Samsung Galaxy S4 M919 16GB T-Mobile 4G LTE Smartphone - Black Mist', 178], ['https://m.media-amazon.com/images/I/41Q3zLdBrUL._AC_UY218_ML3_.jpg', 'Samsung Galaxy J3 2018 (16GB) J337A - 5.0 HD Display, Android 8.0, 4G LTE AT&T Unlocked GSM Smartphone (Black)', 179], ['https://m.media-amazon.com/images/I/61FtFt6rO-L._AC_UY218_ML3_.jpg', 'Samsung Galaxy S8 64GB Unlocked Phone - International Version (Maple Gold)', 180], ['https://m.media-amazon.com/images/I/61Tiv-FndnL._AC_UY218_ML3_.jpg', 'Motorola Moto G7+ Plus (64GB, 4GB RAM) Dual SIM 6.2 4G LTE (GSM Only) Factory Unlocked Smartphone International Model, No Warranty XT1965-2 (Deep Indigo)', 181], ['https://m.media-amazon.com/images/I/61heIFG3R5L._AC_UY218_ML3_.jpg', 'Samsung Galaxy S8 Plus (S8+(SM-G955FD) 4GB RAM / 64GB ROM 6.2-Inch 12MP 4G LTE Dual SIM FACTORY UNLOCKED - International Stock No Warranty (MAPLE GOLD)', 182], ['https://m.media-amazon.com/images/I/81YSPMYJkhL._AC_UY218_ML3_.jpg', 'Samsung Convoy 4 B690 Rugged Water-Resistant Verizon Flip Phone w/ 5MP Camera - Blue', 183], ['https://m.media-amazon.com/images/I/61ufQeEma5L._AC_UY218_ML3_.jpg', 'Apple MGLW2LL/A iPad Air 2 9.7-Inch Retina Display, 16GB, Wi-Fi (Silver) (Renewed)', 184], ['https://m.media-amazon.com/images/I/81+3p0WndhL._AC_UY218_ML3_.jpg', 'Samsung Galaxy Mega 2, Brown Black 16GB (AT&T)', 185], ['https://m.media-amazon.com/images/I/71FVXlimMTL._AC_UY218_ML3_.jpg', 'Xiaomi Redmi 7 32Gb+3GB RAM 6.26 HD+ LTE Factory Unlocked GMS Smartphone (Global Version, No Warranty) (Eclipse Black)', 186], ['https://m.media-amazon.com/images/I/61ve0RjDSUL._AC_UY218_ML3_.jpg', 'Samsung Convoy SCH-U640 Cell Phone Ruggedized PTT 2+ megapixel Camera for Verizon', 187], ['https://m.media-amazon.com/images/I/717jdCOqv6L._AC_UY218_ML3_.jpg', 'Xiaomi Redmi Note 8 Pro (64GB, 6GB) 6.53, 64MP Quad Camera, Helio G90T Gaming Processor, Dual SIM GSM Unlocked - US & Global 4G LTE International Version (Pearl White, 64 GB)', 188], ['https://m.media-amazon.com/images/I/81UgYuadkpL._AC_UY218_ML3_.jpg', 'Xiaomi Redmi Note 8 Pro 64GB, 6GB RAM 6.53 LTE GSM 64MP Factory Unlocked Smartphone - Global Model (Mineral Grey)', 189], ['https://m.media-amazon.com/images/I/61jJeZBliWL._AC_UY218_ML3_.jpg', 'Huawei P30 128GB+6GB RAM (ELE-L29) 6.1 LTE Factory Unlocked GSM Smartphone (International Version) (Black)', 190]]


all_titles = ['Samsung Galaxy Note 3, Black 32GB (Verizon Wireless)', 'Nokia Lumia 900 Black Factory Unlocked', 'Samsung Galaxy Note 9 Factory Unlocked Phone with 6.4 Screen and 128GB (U.S. Warranty),Ocean Blue', 'Samsung Galaxy S7 Edge G935FD 32GB Unlocked GSM 4G LTE', 'Samsung Galaxy Note 5, Black\xa0 64GB (Verizon Wireless)', 'Samsung Galaxy Note 5 SM-N920V Gold 32GB (Verizon Wireless)', 'Motorola G6 – 32 GB – Unlocked (AT&T/Sprint/T-Mobile/Verizon) – Deep Indigo - (U.S. Warranty) - PAAE0011US', 'Moto G - Verizon Prepaid Phone (Verizon Prepaid Only)', 'Samsung Galaxy Note 8 N950 Factory Unlocked Phone 64GB Midnight Black (Renewed)', 'Samsung Galaxy S8+ 64GB Factory Unlocked Smartphone - 6.2 Screen - US Version (Midnight Black) - US Warranty [SM-G955UZKAXAA]', 'Nokia Lumia 635 8GB Unlocked GSM 4G LTE Windows 8.1 Quad-Core Phone - Black', 'Samsung Galaxy S4, White Frost 16GB (AT&T)', 'Samsung Galaxy J5 SM-J500H/DS GSM Factory Unlocked Smartphone, International Version (Gold)', 'Samsung Galaxy S5 G900V Verizon 4G LTE Smartphone w/ 16MP Camera - Black - Verizon', 'Samsung Galaxy A10 32GB (A105M) 6.2 HD+ Infinity-V 4G LTE Factory Unlocked GSM Smartphone - Black', 'Samsung Galaxy A20 US Version Factory Unlocked Cell Phone with 32GB Memory, 6.4 Screen, [SM-A205UZKAXAA], 12 Month Samsung US Warranty,GSM & CDMA Compatible, Black', 'Huawei Mate 10 Pro Unlocked Phone, 6 6GB/128GB, AI Processor, Dual Leica Camera, Water Resistant IP67, GSM Only - Mocha Brown (US Warranty)', 'Samsung Galaxy S7 EDGE G935V 32GB, Verizon/GSM Unlocked, (Renewed) (Black)', 'Samsung Galaxy S7 Edge G935V 32GB Gold - Verizon (Renewed)', 'Samsung Galaxy S4, Black Mist 16GB (Sprint)', 'Motorola DROID Turbo XT1254, Black Ballistic Nylon 32GB (Verizon Wireless)', 'Samsung Galaxy S7 - Black - 32GB - Verizon (Renewed)', 'HUAWEI P20 Lite (32GB + 4GB RAM) 5.84 FHD+ Display, 4G LTE DualSIM GSM Factory Unlocked Smartphone ANE-LX3 - International Model - No Warranty (Sakura Pink)', 'Nokia 6 - 32 GB - Dual Sim Unlocked Smartphone (AT&T/T-Mobile/Metropcs/Cricket/Mint) - Update To Android 9.0 PIE - 5.FHD Screen - Blue - U.S. Warranty', 'Samsung Galaxy Note8 N950U 64GB Unlocked GSM LTE Android Phone w/ Dual 12 Megapixel Camera - Midnight Black', 'Samsung Galaxy A50 A505G 64GB Duos GSM Unlocked Phone w/Triple 25MP Camera - Blue', 'Samsung Galaxy S6 Edge, White Pearl 32GB (Verizon Wireless)', 'Xiaomi Redmi Note 7, 64GB/4GB RAM, 6.30 FHD+, Snapdragon 660, Black - Unlocked Global Version', 'Motorola Droid RAZR M XT907 Verizon Wireless, 8GB, White', 'Nokia Lumia 920, Black 32GB (AT&T)', 'Samsung Galaxy S8 Plus (S8+) (SM-G955FD)4GB RAM / 64GB ROM 6.2-Inch 12MP 4G LTE Dual SIM FACTORY UNLOCKED - International Stock No Warranty (MIDNIGHT BLACK)', 'Nokia 7.1 - Android 9.0 Pie - 64 GB - Dual Camera - Dual SIM Unlocked Smartphone (Verizon/AT&T/T-Mobile/MetroPCS/Cricket/H2O) - 5.84 FHD+ HDR Screen - Steel - U.S. Warranty', 'Samsung Galaxy Note 8 N950U 64GB Unlocked GSM 4G LTE Android Smartphone w/Dual 12 MegaPixel Camera (Renewed) (Midnight Black)', 'Nokia 3310 3G - Unlocked Single SIM Feature Phone (AT&T/T-Mobile/MetroPCS/Cricket/Mint) - 2.4 Inch Screen - Charcoal', 'Sony Xperia XA Ultra (F3213) 4G LTE Unlocked GSM Phone w/ 6 Borderless Display, 21.5MP+16MP Cameras, Octa-Core CPU - White', 'Motorola Moto X (2nd generation) XT1097 GSM Unlocked Cellphone, 16GB, Black Soft Touch', 'Google Pixel XL Phone 32GB - 5.5 inch display ( Factory Unlocked US Version ) (Very Silver)', 'Samsung Galaxy Note 3 N900A 32GB Unlocked GSM Octa-Core Smartphone w/ 13MP Camera - Black', 'Samsung Galaxy A70 128GB/6GB SM-A705M/DS 6.7 HD+ Infinity-U 4G/LTE Factory Unlocked Smartphone (International Version , No Warranty) (Blue)', 'Unlocked GOLD Xiaomi Mi A2, 4GB 64GB, Dual SIM standby, Global Version, 5.5 inch Smartphone Android One, Dual Rear 12.0MP Camera Snapdragon 625', 'Xiaomi Redmi Note 6 Pro 64GB / 4GB RAM 6.26 Dual Camera LTE Factory Unlocked Smartphone Global Version (Black)', 'Google Pixel XL G2PW210032GBBK Factory Unlocked Smartphone, 32GB, 5.5-Inch Display - U.S. Version (Quite Black)', 'Moto G7 Power - Unlocked - 64 GB - Marine Blue (No Warranty) - International Model (GSM Only)', 'Xiaomi Redmi Note 7, 64GB/4GB RAM, 6.30 FHD+, Snapdragon 660, Blue - Unlocked Global Version, No Warranty', 'Samsung Galaxy S7 SM-G930A AT&T Unlocked Smartphone, (Black Onyx)', 'Huawei Mate 20 Lite SNE-LX3 64GB (Factory Unlocked) 6.3 FHD (International Version) (Black)', 'Nokia Lumia 530 White -No Contract (T-Mobile)', 'Samsung Galaxy J7 - Verizon Carrier Locked No Contract Prepaid Smartphone', 'Apple iPhone 7 32GB, Rose Gold (Renewed)', 'Apple iPhone 6S, 64GB, Rose Gold - For AT&T / T-Mobile (Renewed)', 'Apple iPhone X, 256GB, Silver - For AT&T / T-Mobile (Renewed)', 'Apple iPhone X, 256GB, Silver - For AT&T (Renewed)', 'Samsung Galaxy Rugby Pro 4G LTE I547 Unlocked Android Ruggedized Smart Phone', 'Apple iPhone X, Unlocked 5.8, 64GB - Space Gray (Renewed)', 'Samsung Galaxy S4, White Frost 16GB (Sprint)', 'Samsung Focus I917 Unlocked Phone with Windows 7 OS, 5 MP Camera, and Wi-Fi--No Warranty (Black)', 'Nokia Lumia 928, White 32GB (Verizon Wireless)', 'Motorola Barrage V860 Phone (Verizon Wireless)', 'Apple iPhone 6S, 16GB, Rose Gold - For AT&T / T-Mobile (Renewed)', 'Nokia 2 - Android - 8GB - Dual SIM Unlocked Smartphone (AT&T/T-Mobile/MetroPCS/Cricket/H2O) - 5 Screen - Black - U.S. Warranty', 'Samsung Convoy 3, Gray (Verizon Wireless)', 'Samsung Galaxy S6 G920A 32GB Unlocked GSM 4G LTE Octa-Core Android Smartphone with 16MP Camera - Black Sapphire', 'Google Pixel 2 64 GB, Black Factory Unlocked (Renewed)', 'Motorola XT1775 Moto E Plus (4th Gen.) 32GB Unlocked Fine Gold Smartphone', 'Apple iPhone 6S, 64GB, Space Gray - Fully Unlocked (Renewed)', 'Samsung Galaxy Prevail II (Boost Mobile) (Discontinued by Manufacturer)', 'Samsung Galaxy S6 Active, 32 GB , Grey (AT&T)', 'Samsung a157 GoPhone (AT&T)', 'Samsung Galaxy S9 Plus Verizon + GSM Unlocked 64GB Midnight Black (Renewed)', 'Sony Xperia XA1 Ultra G3223 32GB Unlocked GSM LTE Octa-Core Phone w/ 23MP - Gold', 'Google Pixel GSM Unlocked (Renewed) (32GB, Gray)', 'Moto Z GSM Unlocked Smartphone,5.5 Quad HD screen, 64GB storage, 5.2mm thin - Black', 'Moto G7 – Unlocked – 64 GB – Ceramic Black (US Warranty) - Verizon, AT&T, T-Mobile, Sprint, Boost, Cricket, & Metro', 'Google Pixel Phone - 5 inch display (Factory Unlocked US Version) (32GB, Quite Black)', 'Samsung a157V (AT&T Go Phone) No Annual Contract', 'Sony Xperia XA2 Ultra Factory Unlocked Phone - 6 Screen- 32GB - Silver (U.S. Warranty)', 'Motorola XT1585 Turbo 2, (32GB) 5.4 (White, Verizon)', 'Samsung Galaxy Note 5 SM-N920V 32GB White Smartphone for Verizon (Renewed)', 'Samsung Galaxy Note 5 32GB GSM Unlocked - Black (Renewed) (D132)', 'Samsung Galaxy A10 A105M 32GB Duos GSM Unlocked Phone w/ 13MP Camera - Blue', 'Samsung Galaxy S7 Edge 32GB G935A GSM Unlocked (Renewed) (Black)', 'Samsung Galaxy A30 Dual SIM 32GB (SM-A305G/DS) Unlocked Phone GSM International Version - Blue', 'Samsung Galaxy S7 Edge G935A 32GB Gold - Unlocked GSM (Renewed)', 'Samsung Gusto 3, Royal Navy Blue (Verizon Wireless Prepaid) - Discontinued by Manufacturer', 'Motorola DROID MAXX, Black 16GB (Verizon Wireless)', 'Samsung Convoy 3 SCH-U680 Rugged 3G Cell Phone Verizon Wireless', 'Samsung Galaxy Note 10+ Plus Factory Unlocked Cell Phone with 512GB (U.S. Warranty), Aura Black/ Note10+', 'Samsung Galaxy S10 128GB+8GB RAM SM-G973F/DS Dual Sim 6.1LTE Factory Unlocked Smartphone (International Model No Warranty) (Prism White)', 'Google Pixel 2 XL 128 GB, Black (Renewed)', 'Sony Xperia XZ F8332 64GB Forest Blue, 5.2, Dual Sim, GSM Unlocked International Model, No Warranty', 'Verizon Samsung Convoy U660 No Contract Rugged PTT Cell Phone Grey Verizon', 'Sony Xperia XZ F8331 32GB Unlocked GSM 4G LTE Phone w/ 23MP Camera - Mineral Black', 'Samsung Galaxy S10 Factory Unlocked Phone with 128GB - Prism Black', 'Samsung Galaxy S7 32GB G930V Unlocked - Black', 'Samsung Galaxy S7 32GB Unlocked (Verizon Wireless) - Gold', 'Samsung GALAXY S6 G920 32GB Unlocked GSM 4G LTE Octa-Core Smartphone - Black Sapphire', 'Sony Xperia XA1 G3123 32GB Unlocked GSM LTE Octa-Core Phone w/ 23MP Camera - White','Google Pixel 2 XL 64 GB, Black (Renewed)', 'Xiaomi Redmi Note 7 128GB + 4GB RAM 6.3 FHD+ LTE Factory Unlocked 48MP GSM Smartphone (Global Version, No Warranty) (Neptune Blue)', 'Samsung Galaxy Note 4, Frosted White 32GB (Sprint)', 'Xiaomi Pocophone F1 128GB Graphite Black, Dual Sim, 6GB RAM, Dual Camera, 6.18, GSM Unlocked Global Model, No Warranty', 'Xiaomi Mi 8 Lite(64GB, 4GB RAM) 6.26 Full Screen Display, Snapdragon 660, Dual AI Cameras, Factory Unlocked Phone - International Global 4G LTE Version (Black)', 'Samsung Galaxy S7, Black 32GB (Verizon Wireless)', 'Verizon Wireless Motorola RAZR V3m - Silver', 'Samsung Galaxy S7 Edge G935A 32GB AT&T - Black Onyx', 'Samsung Galaxy J2 Core 2018 International Version, No Warranty Factory Unlocked 4G LTE (USA Latin Caribbean) Android Oreo SM-J260M Dual Sim 8MP 8GB (Black)', 'Motorola Droid Turbo - 32GB Android Smartphone - Verizon - Black (Renewed)', 'Sony H3123 - Black/SBH-90C/SCSH10 Xperia XA2 Accessory Bundle, (Bundle Includes: 1 Xperia XA2, 1 SBH90C Bluetooth Headset, 1 Flip Phone Case), Black', 'Sony Xperia X F5121 32GB GSM 23MP Camera Phone - Graphite Black', 'Telcel America Motorola Pre-Paid Cell Phone - Motogo! EX431G - Black', 'Samsung Galaxy A50 US Version Factory Unlocked Cell Phone with 64GB Memory, 6.4 Screen, Black, [SM-A505UZKNXAA]', 'Samsung Galaxy S7 G930T 32GB T-Mobile - Black', 'Samsung Galaxy S7 32GB T-Mobile - Gold Platinum', 'Xiaomi Redmi Note 7 128GB + 4GB RAM 6.3 FHD+ LTE Factory Unlocked 48MP GSM Smartphone (Global Version, No Warranty) (Space Black)', 'Apple iPhone 7, 128GB, Gold - For AT&T / T-Mobile (Renewed)', 'Apple iPhone 6S Plus, 16GB, Silver - For AT&T / T-Mobile (Renewed)', 'Samsung Galaxy S6 G920 32GB Unlocked GSM 4G LTE Octa-Core Smartphone, Gold Platinum', 'Apple iPhone 7, 32GB, Rose Gold - For AT&T / T-Mobile (Renewed)', 'Motorola Moto X4 Factory Unlocked Phone - 32GB - 5.2 Super Black - PA8S0006US', 'Google Pixel 4 - Just Black - 64GB - Unlocked', 'Google Pixel 3 XL 64GB Unlocked GSM & CDMA 4G LTE Android Phone w/ 12.2MP Rear & Dual 8MP Front Camera - Just Black (Renewed)', 'Samsung Galaxy S4 SGH-I337 USA GSM Unlocked Cellphone, 16GB, Frost White', 'Samsung Galaxy S7 Edge, 5.5 32GB (Verizon Wireless) - Blue','Motorola Moto G6 Plus - 64GB - 5.9 FHD+, Dual SIM 4G LTE GSM Factory Unlocked Smartphone International Model XT1926-7 (Deep Indigo)', 'Samsung Galaxy J3 Prime J327A 16GB 4G LTE 7.0 Nougat 5 GSM Unlocked - Black', 'Samsung Galaxy S10e 128GB+6GB RAM SM-G970 Dual Sim 5.8 LTE Factory Unlocked Smartphone (International Model) (Prism Black)', 'Motorola Moto G6 (32GB, 3GB RAM) Dual SIM 5.7 4G LTE (GSM Only) Factory Unlocked Smartphone International Model XT1925-2 (Deep Indigo)', 'Sony Xperia XZ Premium - Unlocked Smartphone - 5.5, 64GB - Dual SIM - Pink (US Warranty)', 'Xiaomi Mi A3 64GB + 4GB RAM, Triple Camera, 4G LTE Smartphone - International Global Version (Not just Blue)', 'Xiaomi Mi A3 64GB, 4GB RAM 6.1 48MP AI Triple Camera LTE Factory Unlocked Smartphone (International Version) (Kind of Grey)', 'Samsung Galaxy J1 Mini 8GB J106H/DS Dual Sim Unlocked Phone - Retail Packaging- Black', 'Apple iPhone Xs Max, 256GB, Space Gray - Fully Unlocked (Renewed)', 'Apple iPhone XS Max, 256GB, Gray - For AT&T (Renewed)', 'Samsung Galaxy S10+ Plus 128GB+8GB RAM SM-G975F/DS Dual Sim 6.4 LTE Factory Unlocked Smartphone International Model, No Warranty (Prism Black)', 'Nokia Lumia Icon, Black 32GB (Verizon Wireless)', 'Motorola Moto One - Android One - 64 GB - 13+2 MP Dual Rear Camera - Dual SIM Unlocked Smartphone (at&T/T-Mobile/MetroPCS/Cricket/H2O) - 5.9 HD+ Display - XT1941-3 - (International Version) (Black)', 'Samsung Galaxy S6 Edge+, Black 64GB (Sprint)', 'Samsung Galaxy S8Active 64GB SM-G892A Unlocked GSM - Meteor Gray (Renewed)', 'Apple iPhone 7 Plus 256GB Unlocked GSM 4G LTE Quad-Core Smartphone - Jet Black (Renewed)', 'Motorola MOTO Z PLAY (XT1635) Factory Unlocked Phone - 5.5 Screen - 32GB - Black (International Version - No Warranty)', 'Samsung Galaxy S6 (SM-G920V) - 32GB Verizon + GSM Smartphone - Black Sapphire (Renewed)', 'Samsung Galaxy S7 Active G891A 32GB Unlocked GSM Shatter,Dust and Water Resistant Smartphone w/ 12MP Camera (AT&T) - Sandy Gold', 'ASUS ZenFone Max Plus ZB570TL-MT67-3G32G-BL - 5.7” 1920x1080-3GB RAM - 32GB storage - LTE Unlocked Dual SIM Cell Phone - US Warranty - Silver', 'Samsung Galaxy S7 G930V 32GB, Verizon, Black Onyx, Unlocked Smartphones (Renewed)', 'Google Pixel 2 128GB Unlocked GSM/CDMA 4G LTE Octa-Core Phone w/ 12.2MP Camera - Just Black', 'Samsung R355C Net 10 Unlimited', 'Google Pixel 2 GSM/CDMA Google Unlocked (Clearly White, 64GB, US warranty)','Google Pixel 2 XL 64GB Unlocked GSM/CDMA 4G LTE Octa-Core Phone w/ 12.2MP Camera - Just Black', 'Samsung Galaxy S6 G920T 32GB Unlocked GSM 4G LTE Octa-Core Android Smartphone w/ 16MP Camera - Black', 'Samsung Galaxy S4 16GB Black SPH-L720T Tri-Band (Boost Mobile)', 'Xiaomi Mi A2 Lite (64GB, 4GB RAM) 5.84 18:9 HD Display, Dual Camera, Android One Unlocked Smartphone - International Global LTE Version (Gold)', 'Samsung Galaxy S6 SM-G920A 32GB White Smartphone for AT&T (Renewed)', 'Samsung Galaxy S4 16GB SPH-L720 4G LTE Android - Sprint (Blue)', 'Apple iPhone XS, 256GB, Gold - Fully Unlocked (Renewed)', 'Apple iPhone XS, 256GB, Gray - For AT&T (Renewed)', 'Samsung Galaxy S10e Factory Unlocked Phone with 128GB, Prism White (Renewed)', 'Apple iPhone 8 Plus 64GB Unlocked GSM Phone - Space Gray (Renewed)', 'Samsung Galaxy S10e Factory Unlocked Phone with 128GB (U.S. Warranty), Prism Blue (Renewed)', 'Samsung Galaxy S6 32GB G920A AT&T Unlocked - Gold Platinum', 'Samsung Galaxy S7 G930P 32GB Gold - Sprint (Renewed)', 'Nokia 105 [2017] TA-1037 Only 2G Dual-Band (850/1900) Factory Unlocked Mobile Phone Black no warranty (White)', 'Huawei Mate 20 Pro LYA-L29 128GB + 6GB - Factory Unlocked International Version - GSM ONLY, NO CDMA - No Warranty in The USA (Black)', 'Apple iPhone 7 Plus 256GB Unlocked GSM Quad-Core Phone - Black (Renewed)', 'Motorola Moto X - 2nd Generation, Black Resin 16GB (Verizon Wireless)', 'Samsung Focus Flash I677 8GB Unlocked GSM Phone with Windows 7.5 OS, 5MP Camera, GPS, Wi-Fi, Bluetooth and FM Radio - Black', 'Nokia 105 RM-1135 Dual-Band (850/1900 MHz) Factory Unlocked Mobile Phone, Black, 2G Network Only.', 'Motorola Moto G4 Play (4th Generation) 16GB 4G LTE Unlocked ONLY GSM 5 Inches International Version No Warranty (White)', 'Samsung Galaxy S6, G920P White Pearl 32GB - Sprint (Renewed)', 'ROG Phone Gaming Smartphone ZS600KL-S845-8G512G - 6 FHD+ 2160x1080 90Hz Display - Qualcomm Snapdragon 845 - 8GB RAM - 512GB Storage - LTE Unlocked Dual SIM Gaming Phone - US Warranty', 'Samsung Galaxy S8 64GB Phone -5.8in Unlocked Smartphone - Midnight Black (Renewed)', 'Sony Xperia X Performance F8131 32GB Unlocked GSM LTE Android Phone w/ 23MP Camera - Black', 'Samsung Galaxy S9 (SM-G960F/DS) 4GB/ 64GB 5.8-inches LTE Dual SIM (GSM Only, No CDMA) Factory Unlocked - International Stock No Warranty (Midnight Black, Phone Only)', 'Samsung Galaxy S8 SM-G950F Unlocked 64GB - International Version/No Warranty (GSM Only, No CDMA) (Midnight Black)', 'Samsung Galaxy Alpha, Frosted Gold 32GB (AT&T)', 'Nokia Lumia 928 32GB Unlocked GSM 4G LTE Windows Smartphone w/ 8MP Carl Zeiss Optics Camera - Black', 'Sony Xperia L1 G3313 16GB Unlocked GSM Quad-Core Android Phone - Pink', 'Apple iPhone X, GSM Unlocked, 256GB - Silver (Renewed)', 'Motorola Moto G6 Play 32GB- Dual SIM 5.7 4G LTE (GSM Only) Factory Unlocked Smartphone International Version XT1922-5 (Deep Indigo)', 'Samsung Galaxy S4 M919 16GB T-Mobile 4G LTE Smartphone - Black Mist', 'Samsung Galaxy J3 2018 (16GB) J337A - 5.0 HD Display, Android 8.0, 4G LTE AT&T Unlocked GSM Smartphone (Black)', 'Samsung Galaxy S8 64GB Unlocked Phone - International Version (Maple Gold)','Motorola Moto G7+ Plus (64GB, 4GB RAM) Dual SIM 6.2 4G LTE (GSM Only) Factory Unlocked Smartphone International Model, No Warranty XT1965-2 (Deep Indigo)', 'Samsung Galaxy S8 Plus (S8+(SM-G955FD) 4GB RAM / 64GB ROM 6.2-Inch 12MP 4G LTE Dual SIM FACTORY UNLOCKED - International Stock No Warranty (MAPLE GOLD)', 'Samsung Convoy 4 B690 Rugged Water-Resistant Verizon Flip Phone w/ 5MP Camera - Blue', 'Apple MGLW2LL/A iPad Air 2 9.7-Inch Retina Display, 16GB, Wi-Fi (Silver) (Renewed)', 'Samsung Galaxy Mega 2, Brown Black 16GB (AT&T)', 'Xiaomi Redmi 7 32Gb+3GB RAM 6.26 HD+ LTE Factory Unlocked GMS Smartphone (Global Version, No Warranty) (Eclipse Black)', 'Samsung Convoy SCH-U640 Cell Phone Ruggedized PTT 2+ megapixel Camera for Verizon', 'Xiaomi Redmi Note 8 Pro (64GB, 6GB) 6.53, 64MP Quad Camera, Helio G90T Gaming Processor, Dual SIM GSM Unlocked - US & Global 4G LTE International Version (Pearl White, 64 GB)', 'Xiaomi Redmi Note 8 Pro 64GB, 6GB RAM 6.53 LTE GSM 64MP Factory Unlocked Smartphone - Global Model (Mineral Grey)', 'Huawei P30 128GB+6GB RAM (ELE-L29) 6.1 LTE Factory Unlocked GSM Smartphone (International Version) (Black)']

def home(request):
    return render(request, 'home.html', {'name': 'Shri'})

def return_title_image_list(li,titles):
    if(len(li) != len(titles)):
        return li

    for i in range(0, len(li)):
        li[i].insert(1, titles[i]) #[['abc.jpg',0],['xyz.jpg',1]....
    return li

def return_image_list(only_image_url):
    for i in only_image_url:
        if(i==''):
            return []
    dl = []
    for i in range(0, len(only_image_url)):
        temp = [only_image_url[i], i]
        dl.append(temp)
    return dl

def all_products(request):
    only_image_url = ['https://m.media-amazon.com/images/I/91eFtaIWpcL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81ZlbLtZ3PL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71VMn6229fL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71ap4Pp+y0L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/814SsOj-45L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/611pE6W+X0L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/51L6DbMbvKL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81VuPb8-arL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/615Y7qMu7lL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/51wJeaY3ekL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61WZWpJTYmL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81suaO+v0mL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61alJun3JvL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/91s1UjlYJHL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/611v1lZLoJL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/51V-XG0uBDL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81a51J6cq9L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/31MTdGW4xHL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/41K7nb5aDBL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71s9fDfT7UL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81VMO0UpxfL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61EzDilvK7L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/819xBtcnz4L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61291x3og8L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71wjfOWQhsL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81SMrSvqUWL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/814WwhaRVuL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61ZaskM5hBL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71-afm8RuLL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61UTfUPGH1L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61MF7kZkrIL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61uRnzVNj9L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/51hvE4YI96L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71iQ4QgMchL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81A-Ww8FqNL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81HfPVitlfL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61EfvRF3sHL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/91Q7P86ef5L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/41cIrncoPlL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61w4AKhyLzL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/517Q3-wHBkL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71PZz7CQ9UL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71nplhIIYjL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/41FBnbqW3pL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/419J7KwTxML._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/51eM8j7wKzL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81ba7gOGt5L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/5103hv5OP0L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/51cRE43zKwL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61+mrwyL24L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81yZOQEC+NL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81yZOQEC+NL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81hxAMIxAeL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/719knfTwPvL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81vLTvMW6DL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81s0xgVWDOL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/41zkjf9ksSL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81k6Nq0KI1L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/41oBClPPoCL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61-rum+PvIL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71Q7aSSyOkL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81oYNE1MUxL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/7108ZFza5gL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71UTxbIjXaL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81s7ZLOGOWL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61NDR6WOqPL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81zlazvfjBL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71Y+FpZYcFL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71SWW5LsZ0L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/817MhreagcL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81o50fc16kL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71se2LK4Y5L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81Vobb06FVL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71z1TjwnadL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/415nK4G4hJL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/719hX34RZhL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61uo-nK+OAL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61MAsozHcaL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81ZGi56SecL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/711v+hjDjxL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61TQfjuS1EL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71i9lcnWT1L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/51H2fS7s9FL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/51ZL33qmCpL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/717w0Z516KL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71Q7aSSyOkL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61Y6BSxzezL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61YVqHdFRxL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71CDE9pG4hL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61XyeFgc3vL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/91xMuzi75uL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71Totr78FmL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/51x8eZ8JbKL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/51i3dF4frhL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71kSYOju0xL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81Z0DX6B3bL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71wchmqQn+L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71CDE9pG4hL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/51X1YcLSmXL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/A1S4AmmN0pL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71q3UGwZhcL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61oKJ6RYCDL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71KB5PiwJAL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61pVtPaTkML._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61y-TkZ5lWL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/51zGEG7p0FL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/91oY6n78hhL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61+387TW4-L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81ZwjKulg8L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/814kh7KdbtL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71kLFOLKN3L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/51C6HtwGL+L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81CgLTDOqQL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/51Hc4HmwwbL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71RYhD1uzpL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61gnQwobQHL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81Fr+G5BcfL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71x3e0x+M2L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/7159JihtggL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61C9GrXEp4L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/41CU2Axt3fL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81suaO+v0mL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71-e3enEyYL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/511i4Qx+e2L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/41RWkB9NDEL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/51BateZ5iqL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71RO-IMRSvL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81nfrNby6HL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/51Xm9ay971L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/41kBtX8-WBL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61Mc+wla27L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71XeQzRDyML._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81nSsCFeiTL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/51V8qIvvd1L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/51W3p2zTv9L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/51Hy0ypovHL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81h0cGu5OcL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61wgvFSAJQL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71c5RZEpLwL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/41YWHpWk7+L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71lpZjjF7NL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/41gYAzWwF1L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71QHUuh-ctL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71RCiZl-V0L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81KgaU7qznL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/51p8LMQ-rIL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/715rN0Y8PqL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81dXcgzgqkL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81cvwVzNoiL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71Bpvs8eUWL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61lfpjOekDL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/614sFnZbspL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61Uy8S7wD9L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/41+2tWGDs3L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61xa10dafvL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61+mDJSfYuL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/810MbmOEoqL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61IR7+oiS7L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/91JHyj8K0FL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71ofnXiUFbL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/41QNnygel4L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61QcWsvnpQL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71b4kAq+5QL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81UpiOZp47L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81O5HRPD1gL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/51X4GhYfx4L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61ktS3pR0TL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61vD0TwZjdL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81qwFH3PTCL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71pjGN8WsoL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71rzaPrNXrL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/51TaayMzqtL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81FWIR3RbUL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81IpIT71b7L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81CFhqN240L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71cLpzYW9IL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81yZOQEC+NL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/51ucn49vPUL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/51hnjqGbVqL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/41Q3zLdBrUL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61FtFt6rO-L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61Tiv-FndnL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61heIFG3R5L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81YSPMYJkhL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61ufQeEma5L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81+3p0WndhL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/71FVXlimMTL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61ve0RjDSUL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/717jdCOqv6L._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/81UgYuadkpL._AC_UY218_ML3_.jpg',
 'https://m.media-amazon.com/images/I/61jJeZBliWL._AC_UY218_ML3_.jpg']
    all_images_urls = return_image_list(only_image_url)
    #print(all_images_urls,"--->",len(all_images_urls))
    all_images_urls = return_title_image_list(all_images_urls, all_titles)
    return render(request, 'products.html', {'image_list': all_images_urls})

#Phone Reccomendation
def product_detail(request):

    item_index = int(request.POST.get('item_index'))
    phone_df = pd.read_csv('items.csv', usecols=['asin', 'image', 'title', 'rating'],
                           dtype={'asin': 'str', 'image': 'str', 'title': 'str', 'rating': 'float32'})
    user_df = pd.read_csv('reviews.csv', usecols=['name', 'asin'], dtype={'name': 'str', 'asin': 'str'})

    df = pd.merge(phone_df, user_df, on='asin')

    combine_phone_rating = df.dropna(axis=0, subset=['asin'])
    phone_ratingCount = (combine_phone_rating.
        groupby(by=['asin'])['rating'].
        count().
        reset_index().
        rename(columns={'rating': 'totalRatingCount'})
    [['asin', 'totalRatingCount']]
        )

    rating_with_totalRatingCount = combine_phone_rating.merge(phone_ratingCount, left_on='asin', right_on='asin',
                                                              how='left')

    rating_with_totalRatingCount.sort_values("totalRatingCount", axis=0, ascending=False,
                                             inplace=True, na_position='last')

    popularity_threshold = 100
    rating_popular_phone = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')

    phone_features_df = rating_popular_phone.pivot_table(index='asin', columns='name', values='rating').fillna(0)

    phone_features_df_matrix = csr_matrix(phone_features_df.values)

    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(phone_features_df_matrix)

    distances, indices = model_knn.kneighbors(phone_features_df.iloc[item_index, :].values.reshape(1, -1),
                                              n_neighbors=8)



    arr = []
    for i in range(0, len(distances.flatten())):
            arr.append('{1}'.format(i, phone_features_df.index[indices.flatten()[i]], distances.flatten()[i]))

    url_arr = []
    reccomended_titles_arr = []
    for s in arr:
        i = phone_df[phone_df['asin'] == s].index[0]
        url_arr.append(phone_df.at[i, 'image'])
        reccomended_titles_arr.append(phone_df.at[i, 'title'])


    return render(request, 'product_details.html', {'recommended_products': url_arr, 'main_item': all_title_images_urls[item_index][0], 'rec_titles_arr': reccomended_titles_arr ,
                                                    'main_title': all_title_images_urls[item_index][1]})

#Phone Price Classification
def opt(request):
    dtrain = pd.read_csv("train.csv")
    dtest = pd.read_csv("test.csv")

    for i in dtrain.columns:
        bser = pd.isnull(dtrain[i])

    X = dtrain.drop(["price_range"], axis=1)
    Y = dtrain["price_range"]

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.1, random_state=1)

    battery = request.POST.get('battery')
    int_mem = request.POST.get('int_mem')
    front_camera = request.POST.get('front_camera')
    rear_camera = request.POST.get('rear_camera')
    cores = request.POST.get('cores')
    wt = request.POST.get('wt')
    ram = request.POST.get('ram')
    dual_sim = request.POST.get('dual_sim')
    fourg = request.POST.get('fourg')
    wifi = request.POST.get('wifi')

    if(battery == '' or int_mem=='' or front_camera=='' or rear_camera=='' or cores=='' or wt=='' or ram=='' or dual_sim=='' or fourg=='' or wifi==''):
        return render(request, 'forminput.html')
    try:
        battery = int(request.POST.get('battery'))
        int_mem = int(request.POST.get('int_mem'))
        front_camera = int(request.POST.get('front_camera'))
        rear_camera = int(request.POST.get('rear_camera'))
        cores = int(request.POST.get('cores'))
        wt = int(request.POST.get('wt'))
        ram = int(request.POST.get('ram'))
        dual_sim = int(request.POST.get('dual_sim'))
        fourg = int(request.POST.get('fourg'))
        wifi = int(request.POST.get('wifi'))
    except ValueError:
        return render(request, 'forminput.html')

    ip = [[battery, dual_sim, front_camera, fourg, int_mem, wt, cores, rear_camera, ram, wifi]]
    #print('*****************', ip)
    xtest = pd.DataFrame(ip, columns=['battery_power', 'dual_sim', 'fc', 'four_g', 'int_memory', 'mobile_wt', 'n_cores',
                                    'pc', 'ram', 'wifi'])

    clf = MultinomialNB()
    clf.fit(xtrain, ytrain)
    pred = clf.predict(xtest)
    result = ''
    if(pred[0] == 0):
        result = 'This is a class D phone'
    elif(pred[0] == 1):
        result = 'This is a class C phone'
    elif (pred[0] == 2):
        result = 'This is a class B phone'
    else:
        result = 'This is a class A phone'

    return render(request, 'opt.html', {'raange': result})

def classify(request):
    return render(request, 'formInput.html')
