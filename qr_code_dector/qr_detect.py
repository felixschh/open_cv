import cv2 as cv


img = cv.imread('qr_code_dector/qr-barcode-card.jpeg')
qr = cv.QRCodeDetector()
x, y, string = qr.detectAndDecode(img)

print(f'Your QR code contains the following Link:\n{x}')