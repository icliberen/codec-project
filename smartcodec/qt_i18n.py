"""Small runtime translator for the desktop UI.

The codec identifiers kept in ``itemData`` never change; only their visible
labels do.  This keeps Turkish/English presentation text from affecting the
actual encode/decode logic.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets


USER_ROLE = int(QtCore.Qt.ItemDataRole.UserRole) + 41


EN_TO_TR = {
    "ycbcr": "YCbCr",
    "AI, ROI and restoration": "AI, ROI ve iyilestirme",
    "Show or hide AI, ROI and restoration settings.": "AI, ROI ve iyilestirme ayarlarini goster veya gizle.",
    "Theme": "Tema", "Language": "Dil", "Light": "Açık", "White": "Beyaz", "Blue": "Mavi", "Dark": "Siyah",
    "Dark mode can be more comfortable for image inspection.": "Karanlık mod, görüntü incelemesini daha rahat hâle getirebilir.",
    "Choose the application language.": "Uygulama dilini seçin.",
    "Choose JPEG or JPEG2000 for a standard image file. DWT/DCT options also expose the transform tree for any loaded image.": "Standart görüntü dosyası için JPEG veya JPEG2000 seçin. DWT/DCT seçenekleri yüklenen her görüntünün dönüşüm ağacını da gösterir.",
    "Lower values create a smaller JPEG file with more visible quality loss. Use 30 for a clear presentation demo.": "Düşük değerler daha küçük JPEG dosyası ve daha görünür kalite kaybı oluşturur. Sunum için 30 kullanılabilir.",
    "Higher rates apply stronger JPEG2000 compression. This setting is used only when JPEG2000 is selected.": "Yüksek oranlar daha güçlü JPEG2000 sıkıştırması uygular. Bu ayar yalnızca JPEG2000 seçiliyken kullanılır.",
    "Controls coefficient quantization: a higher step removes more detail and usually creates a smaller file.": "Katsayı nicemlemesini kontrol eder: yüksek adım daha fazla ayrıntıyı kaldırır ve genellikle daha küçük dosya üretir.",
    "Choose one automatic compression target. BPP and PSNR cannot be active together.": "Tek bir otomatik sıkıştırma hedefi seçin. BPP ve PSNR aynı anda etkin olamaz.",
    "Set the requested encoded bits per image pixel.": "Görüntü pikseli başına istenen sıkıştırılmış bit sayısını belirleyin.",
    "Set the minimum requested reconstruction quality in decibels.": "İstenen en düşük geri oluşturma kalitesini desibel cinsinden belirleyin.",
    "Optional: creates a separate AI-restored image after decoding; it never replaces the actual decoded image.": "İsteğe bağlıdır: geri açmadan sonra ayrı bir AI iyileştirilmiş görüntüsü oluşturur; gerçek çözülmüş görüntünün yerine geçmez.",
    "Draw a rectangle over the Original image to mark an important region.": "Önemli bölgeyi işaretlemek için Orijinal görüntü üzerinde bir dikdörtgen çizin.",
    "Drag on the Original tab to draw an ROI.": "ROI çizmek için Orijinal sekmesinde fareyi sürükleyin.",
    "First select an image in Encode / Decode. Then enable restoration or choose one ROI method below.": "Önce Kodla / Çöz sayfasından bir görüntü seçin. Ardından iyileştirmeyi açın veya aşağıdaki ROI yöntemlerinden birini seçin.",
    "Protect an important area (optional)": "Önemli bir alanı koru (isteğe bağlı)",
    "Draw an area": "Alan çiz",
    "Find objects automatically": "Nesneleri otomatik bul",
    "Find faces": "Yüzleri bul",
    "Choose a mask image": "Maske görüntüsü seç",
    "Use a black-and-white image to specify the important area.": "Önemli alanı belirtmek için siyah-beyaz bir görüntü kullanın.",
    "Clear ROI": "ROI'yi temizle",
    "No ROI selected. The whole image will use the same quality.": "ROI seçilmedi. Tüm görüntü aynı kaliteyi kullanacak.",
    "Advanced ROI settings": "Gelişmiş ROI ayarları",
    "Find people, vehicles and other objects automatically and use them as ROI regions.": "İnsanları, araçları ve diğer nesneleri otomatik bulur ve ROI bölgesi olarak kullanır.",
    "Detect faces and use them as ROI regions.": "Yüzleri algılar ve ROI bölgesi olarak kullanır.",
    "Run compression and decoding together, then fill every comparison panel. This is the recommended presentation button.": "Sıkıştırma ve geri açmayı birlikte çalıştırır, ardından tüm karşılaştırma panellerini doldurur. Sunum için önerilen düğmedir.",
    "Choose how the original and decoded images are displayed.": "Orijinal ve çözülmüş görüntülerin nasıl gösterileceğini seçin.",
    "Choose the image to compare with the original.": "Orijinalle karşılaştırılacak görüntüyü seçin.",
    "Progressive stages": "Aşamalı aşamalar",
    "Transform tree": "Dönüşüm ağacı",
    "Block grid": "Blok ızgarası",
    "Wavelet comparison": "Dalgacık karşılaştırması",
    "db4 / db8 / db12 comparison": "db4 / db8 / db12 karşılaştırması",
    "Fullscreen": "Tam ekran",
    "Fullscreen comparison": "Tam ekran karşılaştırma",
    "Exit fullscreen": "Tam ekrandan çık",
    "Open the comparison in a full-screen window while keeping the image tabs available.": "Görüntü sekmelerini koruyarak karşılaştırmayı tam ekran pencerede açar.",
    "Coefficient / subband preview": "Katsayı / alt bant önizlemesi",
    "Input image": "Kaynak görüntü",
    "DWT": "DWT", "DCT": "DCT",
    "JPEG uses 8×8 DCT blocks; DCT view shows coefficient energy.": "JPEG 8×8 DCT blokları kullanır; DCT görünümü katsayı enerjisini gösterir.",
    "JPEG2000 uses wavelet analysis; this view shows LL/LH/HL/HH subbands.": "JPEG2000 dalgacık analizi kullanır; bu görünüm LL/LH/HL/HH alt bantlarını gösterir.",
    "Each deeper level decomposes the previous LL approximation.": "Her yeni seviye, önceki LL yaklaşım bandını ayrıştırır.",
    "The first tile is one 8×8 coefficient block; other leaves summarize AC energy.": "İlk parça bir 8×8 katsayı bloğudur; diğer yapraklar AC enerjisini özetler.",
    "LL keeps the approximation; LH/HL/HH carry horizontal, vertical and diagonal detail.": "LL yaklaşımı saklar; LH/HL/HH yatay, dikey ve çapraz ayrıntıyı taşır.",
    "Select or show an image to inspect its transform tree.": "Dönüşüm ağacını incelemek için bir görüntü seçin veya gösterin.",
    "Select or show an image to inspect its transform grid.": "Dönüşüm ızgarasını incelemek için bir görüntü seçin veya gösterin.",
    "DCT coefficient energy": "DCT katsayı enerjisi",
    "DWT subband mosaic": "DWT alt bant mozaiği",
    "Every square is one real 8×8 transform block.": "Her kare gerçek bir 8×8 dönüşüm bloğudur.",
    "DCT divides the image into independent 8×8 pixel blocks.": "DCT, görüntüyü bağımsız 8×8 piksel bloklarına ayırır.",
    "DC is at each block's upper-left; AC detail spreads toward its lower-right.": "DC her bloğun sol üstündedir; AC ayrıntıları sağ alta doğru yayılır.",
    "Pixel block": "Piksel bloğu", "Average brightness": "Ortalama parlaklık", "Edges and texture": "Kenarlar ve doku",
    "Each level splits LL into four frequency regions.": "Her seviye LL bandını dört frekans bölgesine ayırır.",
    "The coloured borders show each recursive DWT split on the image plane.": "Renkli sınırlar görüntü düzlemindeki her özyinelemeli DWT ayrımını gösterir.",
    "This is the actual coefficient mosaic produced by the selected DWT settings.": "Bu, seçili DWT ayarlarının ürettiği gerçek katsayı mozaiğidir.",
    "Low-frequency approximation; this region continues to the next level.": "Düşük frekans yaklaşımıdır; sonraki seviyeye bu bölge devam eder.",
    "Horizontal detail coefficients.": "Yatay ayrıntı katsayıları.",
    "Vertical detail coefficients.": "Dikey ayrıntı katsayıları.",
    "Diagonal detail coefficients.": "Çapraz ayrıntı katsayıları.",
    "Approximation / next level": "Yaklaşım / sonraki seviye",
    "Horizontal detail": "Yatay ayrıntı", "Vertical detail": "Dikey ayrıntı", "Diagonal detail": "Çapraz ayrıntı",
    "Band size": "Bant boyutu", "Mean |coefficient|": "Ortalama |katsayı|",
    "Level": "Seviye",
    "Step (higher = more loss)": "Kuantizasyon adımı (yüksek = daha fazla kayıp)",
    "Same image and settings; only the wavelet changes.": "Aynı görüntü ve ayarlar kullanılır; yalnızca dalgacık değişir.",
    "Best quality": "En iyi kalite", "Smallest file": "En küçük dosya",
    "Run the wavelet comparison to see db4, db8 and db12 side by side.": "db4, db8 ve db12 sonuçlarını yan yana görmek için dalgacık karşılaştırmasını çalıştırın.",
    "Wavelet comparison is being processed.": "Dalgacık karşılaştırması işleniyor.",
    "Select or show an image before running the wavelet comparison.": "Dalgacık karşılaştırmasını çalıştırmadan önce bir görüntü seçin veya gösterin.",
    "Wavelet comparison requires lossy mode because lossless mode uses the fixed reversible 5/3 transform.": "Kayıpsız mod sabit tersinir 5/3 dönüşümünü kullandığı için dalgacık karşılaştırması kayıplı modu gerektirir.",
    "The selected image is too small for a db4/db8/db12 comparison.": "Seçili görüntü db4/db8/db12 karşılaştırması için çok küçük.",
    "Original": "Orijinal", "Light compression": "Hafif sıkıştırma", "Medium compression": "Orta sıkıştırma",
    "Lossless transform": "Kayıpsız dönüşüm", "Final result": "Sonuç",
    "Run Encode + Decode to generate progressive stages.": "Aşamalı görüntüleri oluşturmak için Sıkıştır + Geri aç düğmesine basın.",
    "Move the slider to reveal the decoded image.": "Çözülmüş görüntüyü görmek için kaydırıcıyı hareket ettirin.",
    "Resize the comparison image to fit the available area.": "Karşılaştırma görüntüsünü kullanılabilir alana sığdırır.",
    "Leave empty to disable this automatic target.": "Otomatik hedefi devre dışı bırakmak için boş bırakın.",
    "Enter a positive BPP target, or leave empty to disable automatic BPP selection.": "Pozitif bir BPP hedefi girin veya otomatik BPP seçimini kapatmak için boş bırakın.",
    "Enter a PSNR target in dB, or leave empty to disable automatic PSNR selection.": "dB cinsinden bir PSNR hedefi girin veya otomatik PSNR seçimini kapatmak için boş bırakın.",
    "Encode / Decode": "Kodla / Çöz", "Benchmark": "Karşılaştırma", "Video / Transport": "Video / İletim",
    "Smart Codec": "Smart Codec", "JPEG, JPEG2000 and wavelet-based image/video compression platform": "JPEG, JPEG2000 ve dalgacık tabanlı görüntü/video sıkıştırma platformu",
    "Ready. Select an image.": "Hazır. Bir görüntü seçin.", "Cancel": "İptal", "Copy error": "Hatayı kopyala",
    "1. Files": "1. Dosya seçimi", "2. Compression settings": "2. Sıkıştırma seçimi",
    "3. AI, ROI and restoration": "3. Akıllı ROI ve iyileştirme", "4. Actions": "4. İşlem",
    "JPEG/JPEG2000 produces standard files; DWT/DCT/PR-QMF are Smart Codec transform methods.": "JPEG/JPEG2000 standart dosya üretir; DWT/DCT/PR-QMF Smart Codec dönüşüm yöntemleridir.",
    "No model selection: fixed YOLO11m-seg finds ROI objects and the bundled codec-restorer improves details automatically.": "Model seçimi yoktur: sabit YOLO11m-seg ROI nesnelerini bulur, paketli codec-restorer ayrıntıları otomatik iyileştirir.",
    "1. Dosya seçimi": "1. Dosya seçimi", "2. Sıkıştırma seçimi": "2. Sıkıştırma seçimi",
    "3. Akıllı ROI ve iyileştirme": "3. Akıllı ROI ve iyileştirme", "4. İşlem": "4. İşlem",
    "Files": "Dosyalar", "Input image": "Kaynak görüntü", "Encoded output": "Sıkıştırılmış çıktı", "Decoded image": "Geri açılmış görüntü",
    "Kaynak görüntü": "Kaynak görüntü", "Sıkıştırılmış çıktı": "Sıkıştırılmış çıktı", "Geri açılmış görüntü": "Geri açılmış görüntü",
    "Browse": "Seç", "Save": "Kaydet", "Show image": "Görüntüyü göster", "Encode": "Sıkıştır", "Decode": "Geri aç",
    "Encode + Decode": "Sıkıştır + karşılaştır", "Batch": "Klasör işlemi", "Görüntüyü göster": "Görüntüyü göster",
    "Sıkıştır": "Sıkıştır", "Geri aç": "Geri aç", "Sıkıştır + karşılaştır": "Sıkıştır + karşılaştır", "Klasör işlemi": "Klasör işlemi",
    "Codec settings": "Sıkıştırma ayarları", "Profile": "Hazır ayar", "Method": "Çıktı formatı", "Output format": "Çıktı formatı",
    "Hazır ayar": "Hazır ayar", "Çıktı formatı": "Çıktı formatı", "Mode": "Kodlama türü", "Kodlama türü": "Kodlama türü",
    "Custom": "Özel", "High quality": "Yüksek kalite", "Maximum compression": "Maksimum sıkıştırma",
    "Lossless medical": "Kayıpsız tıbbi", "ROI object": "ROI nesnesi", "lossy": "Kayıplı", "lossless": "Kayıpsız",
    "JPEG quality": "JPEG kalite", "JPEG quality (1-100)": "JPEG kalite (1-100)", "JPEG kalite (1-100)": "JPEG kalite (1-100)",
    "JPEG2000 rate": "JPEG2000 oranı", "JPEG2000 oranı": "JPEG2000 oranı", "Wavelet": "Dalgacık", "Dalgacık": "Dalgacık",
    "DWT level": "DWT seviye", "DWT seviye": "DWT seviye", "Step": "Kuantizasyon adımı", "Kuantizasyon adımı": "Kuantizasyon adımı",
    "Quantizer": "Kuantizer", "Kuantizer": "Kuantizer", "Color space": "Renk uzayı", "Renk uzayı": "Renk uzayı",
    "Quality target": "Kalite hedefi", "Kalite hedefi": "Kalite hedefi",
    "Target BPP": "Hedef BPP", "Hedef BPP": "Hedef BPP", "Target PSNR": "Hedef PSNR", "Hedef PSNR": "Hedef PSNR",
    "Wavelet-based Smart Codec settings": "Dalgacık tabanlı Smart Codec ayarları", "Dalgacık tabanlı Smart Codec ayarları": "Dalgacık tabanlı Smart Codec ayarları",
    "AI and ROI": "Akıllı ROI ve iyileştirme", "Use AI restoration (TorchScript)": "Sıkıştırma sonrası detay iyileştirme",
    "Sıkıştırma sonrası detay iyileştirme": "Sıkıştırma sonrası detay iyileştirme", "AI model": "AI modeli", "Find best local AI model": "Yerel AI modelini bul",
    "ROI boxes": "ROI kutuları", "ROI kutuları": "ROI kutuları", "ROI mask": "ROI maskesi", "ROI maskesi": "ROI maskesi",
    "ROI strength": "ROI önceliği", "ROI önceliği": "ROI önceliği", "Feather": "Yumuşatma", "Yumuşatma": "Yumuşatma",
    "Draw ROI": "ROI çiz", "ROI çiz": "ROI çiz", "Detect YOLO ROI": "AI ile nesneleri bul", "AI ile nesneleri bul": "AI ile nesneleri bul",
    "Detect face ROI": "Yüz ROI bul", "Yüz ROI bul": "Yüz ROI bul", "Clear": "Temizle", "Temizle": "Temizle",
    "Original": "Orijinal", "Decoded": "Çözülmüş", "Restored": "AI iyileştirilmiş", "Difference": "Fark", "Compare": "Karşılaştır",
    "View": "Görünüm", "Compare with": "Karşılaştırılan", "Slider": "Kaydırıcı", "Side-by-side": "Yan yana",
    "Error heatmap": "Hata ısı haritası", "Histogram": "Histogram", "Fit": "Sığdır", "Compression ratio": "Sıkıştırma oranı",
    "PSNR–BPP graph": "PSNR–BPP grafiği",
    "Same image, shared BPP targets; dots show achieved BPP and PSNR.": "Aynı görüntü, ortak BPP hedefleri; noktalar ölçülen BPP ve PSNR'ı gösterir.",
    "Larger markers: reference BPP target": "Büyük noktalar: referans BPP hedefi",
    "Target not reached; the plotted BPP is the actual file rate.": "Hedefe ulaşılamadı; grafikte gerçek dosya BPP değeri gösterilir.",
    "8-bit grayscale": "8-bit gri seviye",
    "256 gray levels • 0 = black • 255 = white • 8 bits/pixel": "256 gri ton • 0 = siyah • 255 = beyaz • 8 bit/piksel",
    "Preview only: files and compression are unchanged. Metrics above describe the original encode/decode result.": "Yalnızca önizleme: dosyalar ve sıkıştırma değişmez. Üstteki metrikler asıl kodlama/çözme sonucuna aittir.",
    "Gray = round(0.299R + 0.587G + 0.114B). No automatic contrast stretching.": "Gri = yuvarla(0,299R + 0,587G + 0,114B). Otomatik kontrast genişletme uygulanmaz.",
    "16-bit input: 0–65535 maps to 0–255 for this preview only.": "16-bit kaynak: 0–65535 aralığı yalnızca bu önizleme için 0–255 aralığına dönüştürülür.",
    "Measured result": "Ölçüm sonucu",
    "Each point is one measured result: BPP (bottom), PSNR (left).": "Her nokta bir ölçüm sonucudur: BPP (alt), PSNR (sol).",
    "Compression trial": "Sıkıştırma denemesi",
    "Real encode–decode trials. Independent scales: BPP (left), PSNR (right).": "Gerçek kodlama–çözme denemeleri. Ayrı ölçekler: BPP (sol), PSNR (sağ).",
    "∞ = exact reconstruction": "∞ = kayıpsız geri oluşturma",
    "Run Encode + Decode to compare this image.": "Bu görüntüyü karşılaştırmak için Sıkıştır + karşılaştır işlemini çalıştırın.",
    "Measured rate–distortion curve from real encode–decode results.": "Gerçek kodlama–çözme sonuçlarından ölçülen hız–bozulma eğrisi.",
    "Bits per pixel (BPP)": "Piksel başına bit (BPP)",
    "Current result": "Geçerli sonuç", "Best measured quality": "Ölçülen en iyi kalite",
    "Run Encode + Decode to generate the PSNR–BPP graph.": "PSNR–BPP grafiğini oluşturmak için Sıkıştır + karşılaştır işlemini çalıştırın.",
    "PSNR–BPP graph is being processed.": "PSNR–BPP grafiği işleniyor.",
    "PSNR–BPP graph requires lossy mode.": "PSNR–BPP grafiği kayıplı kodlama modu gerektirir.",
    "Original size": "İlk boyut", "Encoded size": "Son boyut",
    "Luminance histogram": "Parlaklık histogramı", "Intensity": "Yoğunluk",
    "Normalized pixel count": "Normalize piksel sayısı", "Mean": "Ortalama", "Std dev": "Std. sapma",
    "Peak intensity": "Tepe yoğunluğu",
    "Original and decoded luminance distribution across 64 intensity bins.": "Orijinal ve çözülmüş parlaklık dağılımı, 64 yoğunluk aralığında gösterilir.",
    "No image": "Görüntü yok", "No encode/decode operation yet.": "Henüz kodlama/çözme işlemi yapılmadı.",
    "AI restoration is disabled. Enable it to create a restored image.": "AI iyileştirme kapalı. İyileştirilmiş görüntü üretmek için etkinleştirin.",
    "Benchmark settings": "Karşılaştırma ayarları", "Input folder": "Girdi klasörü", "Output folder": "Çıktı klasörü",
    "Choose an image folder and what you want to compare. Recommended settings are ready for first use.": "Bir görüntü klasörü ve neyi karşılaştırmak istediğinizi seçin. Önerilen ayarlar ilk kullanım için hazırdır.",
    "What do you want to compare?": "Neyi karşılaştırmak istiyorsunuz?",
    "Quick overall comparison": "Hızlı genel karşılaştırma",
    "Compare at the same quality (PSNR)": "Aynı kalitede karşılaştır (PSNR)",
    "Compare at the same file size (BPP)": "Aynı dosya boyutunda karşılaştır (BPP)",
    "Compare ROI on and off": "ROI açık ve kapalı karşılaştır",
    "Target value": "Hedef değer", "Target PSNR (dB)": "Hedef PSNR (dB)",
    "Create example images": "Örnek görüntüler oluştur", "Run comparison": "Karşılaştırmayı çalıştır",
    "Open result chart": "Sonuç grafiğini aç", "Advanced benchmark settings": "Gelişmiş karşılaştırma ayarları",
    "Step values": "Adım değerleri", "Wavelets": "Dalgacıklar", "Normalization": "Normalleştirme", "Target PSNR/BPP": "Hedef PSNR/BPP",
    "Allocation": "Bit dağıtımı", "Generate sample images": "Örnek görüntüler üret", "Generate category dataset": "Kategori veri seti üret",
    "Run benchmark": "Karşılaştırmayı çalıştır", "Open chart": "Grafiği aç", "Category": "Kategori", "Image": "Görüntü", "Status": "Durum",
    "Video files": "Video dosyaları", "Input video": "Kaynak video", "Frame output folder": "Kare çıktı klasörü", "Manifest": "Manifest",
    "Simple workflow: choose a video, encode its frames, then create the decoded MP4. Transport testing is optional.": "Basit akış: bir video seçin, karelerini kodlayın ve ardından çözülmüş MP4'ü oluşturun. İletim testi isteğe bağlıdır.",
    "Compression method": "Sıkıştırma yöntemi", "Compression strength": "Sıkıştırma gücü",
    "Higher values create smaller files with more visible quality loss.": "Yüksek değerler daha küçük dosya ve daha görünür kalite kaybı oluşturur.",
    "Higher compression strength usually means a smaller file and more visible distortion.": "Daha yüksek sıkıştırma gücü genellikle daha küçük dosya ve daha görünür bozulma anlamına gelir.",
    "Advanced video settings": "Gelişmiş video ayarları",
    "After encoding, the manifest is filled automatically. You can also select an existing manifest.": "Kodlamadan sonra manifest otomatik doldurulur. İsterseniz mevcut bir manifest de seçebilirsiniz.",
    "Create decoded MP4": "Çözülmüş MP4 oluştur",
    "Decoded video": "Çözülmüş video", "Video codec and motion": "Video kodlama ve hareket", "Motion": "Hareket",
    "Video ROI mask": "Video ROI maskesi", "Motion compensation": "Hareket telafisi", "ROI motion tracking": "ROI hareket takibi",
    "Encode video frames": "Video karelerini kodla", "Decode manifest to video": "Manifesti videoya çöz",
    "Transport simulation": "İletim simülasyonu", "Semantic transport bands": "Anlamsal iletim bantları", "Spatial tiles": "Bölgesel parçalar",
    "This test shows what happens when parts of the compressed file are lost during transmission.": "Bu test, sıkıştırılmış dosyanın bazı parçaları iletim sırasında kaybolduğunda ne olduğunu gösterir.",
    "Packet loss (0.10 = 10%)": "Paket kaybı (0,10 = %10)", "Advanced transport settings": "Gelişmiş iletim ayarları",
    "Frame-level partial transport": "Kare bazlı kısmi iletim", "Tile px": "Parça pikseli", "Packet loss": "Paket kaybı",
    "Transport output": "İletim çıktısı", "Backend": "Altyapı", "UDP target host": "UDP hedef sunucusu", "UDP port": "UDP portu",
    "Live UDP XOR-FEC": "Canlı UDP XOR-FEC", "Run transport": "İletimi çalıştır",
    "simulation": "simülasyon", "live-udp": "canlı-udp", "none": "yok", "translation": "öteleme", "block": "blok", "optical-flow": "optik-akış",
}

TR_TO_EN = {tr: en for en, tr in EN_TO_TR.items() if en != tr}


def canonical(text: str) -> str:
    # Older saved sessions may still contain the former Turkish Dark label.
    if text == "Koyu":
        return "Dark"
    return TR_TO_EN.get(text, text)


def translate(text: str, language: str) -> str:
    source = canonical(text)
    if source == "ycbcr":
        return "YCbCr"
    return EN_TO_TR.get(source, source) if language == "tr" else source


def translate_status(message: str, language: str) -> str:
    """Translate runtime status text, including messages containing paths or values.

    Static widgets can be translated as a whole. Status messages often contain
    an output path, a counter, or a metric, so they need small phrase-level
    replacements instead.
    """
    if language != "tr":
        return canonical(message)
    translated = canonical(message)
    phrases = (
        ("Encoding and comparison complete; bundled restoration model used: ", "Sıkıştırma ve karşılaştırma tamamlandı; paketli iyileştirme modeli kullanıldı: "),
        ("Encoding and comparison complete; the restoration model was unavailable, so the baseline method was used.", "Sıkıştırma ve karşılaştırma tamamlandı; iyileştirme modeli kullanılamadığı için temel yöntem uygulandı."),
        ("Encoding and comparison complete.", "Sıkıştırma ve karşılaştırma tamamlandı."),
        ("Encoding complete. Press Save to keep the encoded output.", "Sıkıştırma tamamlandı. Çıktıyı kalıcı tutmak için Kaydet'e basın."),
        ("Encoding complete. The result is temporary until Save is pressed.", "Sıkıştırma tamamlandı. Kaydet'e basılana kadar sonuç geçicidir."),
        ("Decoding complete. The image is temporary until Save is pressed.", "Geri açma tamamlandı. Kaydet'e basılana kadar görüntü geçicidir."),
        ("Encoded output saved: ", "Sıkıştırılmış çıktı kaydedildi: "),
        ("Decoded image saved: ", "Geri açılmış görüntü kaydedildi: "),
        ("Encoding complete: ", "Sıkıştırma tamamlandı: "),
        ("Decoding complete: ", "Geri açma tamamlandı: "),
        ("Batch encoding complete: ", "Toplu sıkıştırma tamamlandı: "),
        ("Batch decoding complete: ", "Toplu geri açma tamamlandı: "),
        ("Image loaded: ", "Görüntü yüklendi: "),
        ("ROI added: ", "ROI eklendi: "),
        ("ROI cleared.", "ROI temizlendi."),
        ("Lossless mode uses the reversible integer 5/3 wavelet.", "Kayıpsız mod, tersinir tamsayı 5/3 dalgacığını kullanır."),
        ("Lossy mode supports ROI prioritization and detail restoration.", "Kayıplı mod, ROI önceliklendirme ve detay iyileştirmeyi destekler."),
        ("Error: ", "Hata: "),
        ("Working: ", "Çalışıyor: "),
        ("Operation cancelled after the current step.", "İşlem geçerli adımın ardından iptal edildi."),
        ("Cancellation requested; the active step will finish safely.", "İptal istendi; etkin adım güvenle tamamlanacak."),
        ("Error details copied to the clipboard.", "Hata ayrıntıları panoya kopyalandı."),
        ("Image loaded. You can now encode or decode.", "Görüntü yüklendi. Şimdi sıkıştırabilir veya geri açabilirsiniz."),
        ("Waiting for encoding to finish…", "Sıkıştırmanın tamamlanması bekleniyor…"),
        ("Encoding complete. File: ", "Sıkıştırma tamamlandı. Dosya: "),
        ("Size: ", "Boyut: "),
        ("Original size: ", "İlk boyut: "),
        ("Encoded size: ", "Son boyut: "),
        ("Compression: ", "Sıkıştırma: "),
        ("Target BPP: ", "Hedef BPP: "),
        ("Achieved BPP: ", "Gerçekleşen BPP: "),
        ("Target PSNR: ", "Hedef PSNR: "),
        ("Achieved PSNR: ", "Gerçekleşen PSNR: "),
        ("Target error: ", "Hedef sapması: "),
        ("Restored MSE: ", "İyileştirilmiş MSE: "),
        ("Background PSNR: ", "Arka plan PSNR: "),
        ("Preset saved: ", "Hazır ayar kaydedildi: "),
        ("Preset loaded: ", "Hazır ayar yüklendi: "),
        ("Wavelet comparison complete: db4, db8 and db12.", "Dalgacık karşılaştırması tamamlandı: db4, db8 ve db12."),
        ("PSNR–BPP graph complete.", "PSNR–BPP grafiği tamamlandı."),
    )
    for english, turkish in phrases:
        translated = translated.replace(english, turkish)
    if translated.endswith(" started…"):
        operation = translated.removesuffix(" started…")
        return f"{translate(operation, 'tr')} işlemi başlatıldı…"
    return translated


def combo_value(combo: QtWidgets.QComboBox) -> str:
    value = combo.currentData(USER_ROLE)
    return str(value) if value is not None else canonical(combo.currentText())


def set_combo_value(combo: QtWidgets.QComboBox, value: object) -> None:
    wanted = str(value)
    index = combo.findData(wanted, USER_ROLE)
    if index >= 0:
        combo.setCurrentIndex(index)
        return
    # Before the first language pass, combo items do not yet have canonical
    # USER_ROLE values.  Match their source text case-insensitively as well;
    # this matters for values such as ``ycbcr`` whose display spelling is
    # ``YCbCr`` after translation.
    wanted_canonical = canonical(wanted).casefold()
    translated = translate(wanted, str(combo.property("ui_language") or "en"))
    for candidate in range(combo.count()):
        item_data = combo.itemData(candidate, USER_ROLE)
        item_value = str(item_data) if item_data is not None else canonical(combo.itemText(candidate))
        if item_value.casefold() == wanted_canonical or combo.itemText(candidate).casefold() == translated.casefold():
            combo.setCurrentIndex(candidate)
            return
    combo.setCurrentText(translated)


def apply_language(root: QtWidgets.QWidget, language: str) -> None:
    """Translate static widget text while preserving combo-box machine values."""
    root.setProperty("ui_language", language)
    for widget in [root, *root.findChildren(QtWidgets.QWidget)]:
        widget.setProperty("ui_language", language)
        source_tooltip = widget.property("ui_tooltip_source")
        if source_tooltip is None and widget.toolTip():
            source_tooltip = canonical(widget.toolTip())
            widget.setProperty("ui_tooltip_source", source_tooltip)
        if source_tooltip:
            widget.setToolTip(translate(str(source_tooltip), language))
        if isinstance(widget, QtWidgets.QComboBox) and widget.objectName() != "languageCombo":
            current = combo_value(widget)
            for index in range(widget.count()):
                source = widget.itemData(index, USER_ROLE)
                if source is None:
                    source = canonical(widget.itemText(index))
                    widget.setItemData(index, source, USER_ROLE)
                widget.setItemText(index, translate(str(source), language))
            set_combo_value(widget, current)
        elif isinstance(widget, QtWidgets.QGroupBox):
            widget.setTitle(translate(widget.title(), language))
        elif isinstance(widget, QtWidgets.QAbstractButton):
            widget.setText(translate(widget.text(), language))
        elif isinstance(widget, QtWidgets.QLabel):
            source_text = widget.property("source_text")
            if source_text is not None:
                widget.setText(translate_status(str(source_text), language))
            else:
                widget.setText(translate(widget.text(), language))
    for tabs in [root, *root.findChildren(QtWidgets.QTabWidget)]:
        if not isinstance(tabs, QtWidgets.QTabWidget):
            continue
        for index in range(tabs.count()):
            source = tabs.tabBar().tabData(index)
            if source is None:
                source = canonical(tabs.tabText(index))
                tabs.tabBar().setTabData(index, source)
            tabs.setTabText(index, translate(str(source), language))
