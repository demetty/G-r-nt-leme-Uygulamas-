import cv2
import numpy as np
from tkinter import Tk, filedialog

def dosya_sec():
    root = Tk()
    root.withdraw()
    dosya = filedialog.askopenfilename(
        title="Görüntü Seç",
        filetypes=(("Görüntü Dosyaları", "*.jpg *.jpeg *.png *.bmp"), ("Tüm Dosyalar", "*.*"))
    )
    root.destroy()
    return dosya

def dosya_kaydet():
    root = Tk()
    root.withdraw()
    dosya = filedialog.asksaveasfilename(
        title="Görüntüyü Kaydet",
        defaultextension=".jpg",
        filetypes=(("JPEG", "*.jpg"), ("PNG", "*.png"), ("BMP", "*.bmp"), ("Tüm Dosyalar", "*.*"))
    )
    root.destroy()
    return dosya

def menu():
    print("""
    Gelişmiş Resim İşleme Programı
    1. Görüntüyü Gri Tonlama
    2. Gaussian Bulanıklaştırma
    3. Kenar Tespiti (Canny)
    4. Görüntüyü Döndürme
    5. Görüntü Boyutlandırma
    6. Görüntüyü Kırpma
    7. Görüntü Erozyonu
    8. Görüntü Dilasyonu
    9. Görüntüye Metin Ekleme
    10. Renk Eşitleme
    11. Sobel Kenar Algılama
    12. Işıldama Efekti
    13. Kontrast Artırma
    14. Keskinleştirme
    15. Sepya Efekti
    16. Görüntüyü Kaydetme
    17. Orijinal Görüntüye Dön
    0. Çıkış
    """)

def gri_tonlama(image):
    if len(image.shape) == 2:  # Zaten gri tonlamalıysa
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def bulaniklastirma(image):
    return cv2.GaussianBlur(image, (15, 15), 0)

def kenar_tespiti(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return cv2.Canny(gray, 100, 200)

def dondurme(image, derece):
    (h, w) = image.shape[:2]
    merkez = (w // 2, h // 2)
    matris = cv2.getRotationMatrix2D(merkez, derece, 1.0)
    return cv2.warpAffine(image, matris, (w, h))

def boyutlandirma(image, oran):
    if oran <= 0:
        raise ValueError("Oran 0'dan büyük olmalıdır!")
    yeni_boyut = (int(image.shape[1] * oran), int(image.shape[0] * oran))
    return cv2.resize(image, yeni_boyut)

def kirpma(image, x1, y1, x2, y2):
    h, w = image.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
        raise ValueError("Geçersiz kırpma koordinatları!")
    return image[y1:y2, x1:x2]

def erozyon(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

def dilasyon(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def metin_ekle(image, text, x, y):
    img_copy = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_copy, text, (x, y), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    return img_copy

def renk_esitleme(image):
    if len(image.shape) == 2:  # Gri tonlamalı görüntü
        return cv2.equalizeHist(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def sobel_kenar(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)
    return np.uint8(np.absolute(sobel_edges))

def isildama_efekti(image):
    blurred = cv2.GaussianBlur(image, (21, 21), 0)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

def kontrast_artirma(image, alpha=2.0, beta=50):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def keskinlestirme(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def sepya_efekti(image):
    if len(image.shape) == 2:  # Gri tonlamalı görüntü
        return image
    img_sepia = np.array(image, dtype=np.float64)
    img_sepia = cv2.transform(img_sepia, np.array([[0.393, 0.769, 0.189], 
                                                  [0.349, 0.686, 0.168], 
                                                  [0.272, 0.534, 0.131]]))
    img_sepia = np.clip(img_sepia, 0, 255)
    return np.uint8(img_sepia)

def kaydet(image, dosya_yolu):
    try:
        if dosya_yolu:
            cv2.imwrite(dosya_yolu, image)
            print(f"Görüntü başarıyla kaydedildi: {dosya_yolu}")
            return True
        return False
    except Exception as e:
        print(f"Kaydetme hatası: {e}")
        return False

def main():
    try:
        print("Lütfen bir görüntü dosyası seçin...")
        image_path = dosya_sec()
        
        if not image_path:
            print("Dosya seçilmedi!")
            return
            
        original_image = cv2.imread(image_path)
        if original_image is None:
            print("Görüntü yüklenemedi!")
            return

        current_image = original_image.copy()
        
        while True:
            menu()
            secimler = input("Birden fazla işlem yapmak için numaraları virgülle ayırarak girin (ör. 1,2,5): ").strip()
            
            if not secimler:
                print("Geçerli bir seçim yapınız!")
                continue
                
            secimler_listesi = [s.strip() for s in secimler.split(',')]
            
            try:
                for secim in secimler_listesi:
                    if secim == "1":
                        current_image = gri_tonlama(current_image)
                    elif secim == "2":
                        current_image = bulaniklastirma(current_image)
                    elif secim == "3":
                        current_image = kenar_tespiti(current_image)
                    elif secim == "4":
                        derece = float(input("Döndürme derecesi girin: "))
                        current_image = dondurme(current_image, derece)
                    elif secim == "5":
                        oran = float(input("Boyutlandırma oranı girin (ör. 0.5): "))
                        current_image = boyutlandirma(current_image, oran)
                    elif secim == "6":
                        print(f"Görüntü boyutları: {current_image.shape[1]}x{current_image.shape[0]}")
                        x1 = int(input("Kırpma başlangıç x koordinatı: "))
                        y1 = int(input("Kırpma başlangıç y koordinatı: "))
                        x2 = int(input("Kırpma bitiş x koordinatı: "))
                        y2 = int(input("Kırpma bitiş y koordinatı: "))
                        current_image = kirpma(current_image, x1, y1, x2, y2)
                    elif secim == "7":
                        current_image = erozyon(current_image)
                    elif secim == "8":
                        current_image = dilasyon(current_image)
                    elif secim == "9":
                        metin = input("Görüntüye eklemek istediğiniz metin: ")
                        print(f"Görüntü boyutları: {current_image.shape[1]}x{current_image.shape[0]}")
                        x = int(input("Metnin x koordinatını girin: "))
                        y = int(input("Metnin y koordinatını girin: "))
                        current_image = metin_ekle(current_image, metin, x, y)
                    elif secim == "10":
                        current_image = renk_esitleme(current_image)
                    elif secim == "11":
                        current_image = sobel_kenar(current_image)
                    elif secim == "12":
                        current_image = isildama_efekti(current_image)
                    elif secim == "13":
                        current_image = kontrast_artirma(current_image)
                    elif secim == "14":
                        current_image = keskinlestirme(current_image)
                    elif secim == "15":
                        current_image = sepya_efekti(current_image)
                    elif secim == "16":
                        dosya_yolu = dosya_kaydet()
                        if kaydet(current_image, dosya_yolu):
                            print("İşleme devam etmek istiyor musunuz? (E/H)")
                            if input().upper() != 'E':
                                return
                    elif secim == "17":
                        current_image = original_image.copy()
                        print("Orijinal görüntüye dönüldü.")
                    elif secim == "0":
                        print("Programdan çıkılıyor...")
                        return
                    else:
                        print(f"Geçersiz seçim: {secim}")
                        continue

                cv2.imshow("İşlenmiş Görüntü", current_image)
                key = cv2.waitKey(0)
                cv2.destroyAllWindows()

            except ValueError as ve:
                print(f"Değer hatası: {ve}")
                continue
            except Exception as e:
                print(f"Hata oluştu: {e}")
                continue

    except KeyboardInterrupt:
        print("\nProgram kullanıcı tarafından sonlandırıldı.")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
