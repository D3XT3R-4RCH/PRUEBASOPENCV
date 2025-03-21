import cv2
import numpy as np
import pytesseract

# Especifica la ruta al ejecutable de Tesseract (ajusta si necesario)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Variables globales para los parámetros ajustables
bilateral_d = 11
bilateral_sigma_color = 17
bilateral_sigma_space = 17
canny_min = 30
canny_max = 200
approx_factor = 0.018

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
    edged = cv2.Canny(gray, canny_min, canny_max)
    return gray, edged

def find_plate_contour(edged, image):
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    plate_contour = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, approx_factor * peri, True)
        if len(approx) == 4:
            plate_contour = approx
            break
    
    return plate_contour

def update_image(val):
    global bilateral_d, bilateral_sigma_color, bilateral_sigma_space, canny_min, canny_max, approx_factor
    
    # Obtener valores de las trackbars
    bilateral_d = cv2.getTrackbarPos('Bilateral D', 'Ajustes') * 2 + 1  # Asegurar que sea impar
    bilateral_sigma_color = cv2.getTrackbarPos('Sigma Color', 'Ajustes')
    bilateral_sigma_space = cv2.getTrackbarPos('Sigma Space', 'Ajustes')
    canny_min = cv2.getTrackbarPos('Canny Min', 'Ajustes')
    canny_max = cv2.getTrackbarPos('Canny Max', 'Ajustes')
    approx_factor = cv2.getTrackbarPos('Approx Factor', 'Ajustes') / 1000.0  # Escala fina
    
    # Procesar la imagen con los nuevos valores
    image_copy = image.copy()
    gray, edged = preprocess_image(image_copy)
    plate_contour = find_plate_contour(edged, image_copy)
    
    if plate_contour is not None:
        cv2.drawContours(image_copy, [plate_contour], -1, (0, 255, 0), 3)
    
    # Mostrar las imágenes
    cv2.imshow('Imagen con contorno', image_copy)
    cv2.imshow('Bordes', edged)

def main(image_path):
    global image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: No se pudo cargar la imagen.")
        return
    
    # Crear ventana de ajustes
    cv2.namedWindow('Ajustes')
    
    # Crear trackbars para ajustar parámetros
    cv2.createTrackbar('Bilateral D', 'Ajustes', bilateral_d // 2, 20, update_image)  # 1-41 (impar)
    cv2.createTrackbar('Sigma Color', 'Ajustes', bilateral_sigma_color, 100, update_image)
    cv2.createTrackbar('Sigma Space', 'Ajustes', bilateral_sigma_space, 100, update_image)
    cv2.createTrackbar('Canny Min', 'Ajustes', canny_min, 255, update_image)
    cv2.createTrackbar('Canny Max', 'Ajustes', canny_max, 255, update_image)
    cv2.createTrackbar('Approx Factor', 'Ajustes', int(approx_factor * 1000), 50, update_image)  # 0.001-0.05
    
    # Mostrar la imagen inicial
    update_image(0)
    
    # Esperar hasta que se presione 'q' para salir
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = r"C:\Users\PC\Desktop\PruebasOpenCV\img\placaplaca.jpg"
    main(image_path)