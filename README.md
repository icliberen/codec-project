# Smart Codec

**Görüntü sıkıştırmayı yalnızca ölçmek değil, görerek anlamak için.**

Smart Codec; DCT, DWT, JPEG ve JPEG2000 yöntemlerini aynı görüntü üzerinde deneyebileceğiniz, Python ve PySide6 ile geliştirilmiş bir eğitim ve sunum uygulamasıdır. Türkçe/İngilizce arayüzü, White/Blue/Dark temaları ve tam ekran karşılaştırma araçları bulunur.

> Bu bir eğitim/araştırma projesidir. Özel `.swc` biçimi JPEG veya JPEG2000 standardı değildir. Standart `.jpg` ve `.jp2` çıktıları ayrı Pillow/OpenJPEG yolu üzerinden üretilir. Klinik veya kritik sistemler için doğrulanmamıştır.

## Windows'ta çalıştırma

1. [Releases](https://github.com/icliberen/codec-project/releases) sayfasından **Smart-Codec-Windows-x64.zip** paketini indirin.
2. ZIP'in **tamamını**, yazma izniniz olan bir klasöre çıkarın.
3. `Smart Codec/Smart Codec.exe` dosyasını açın. Python kurmanız gerekmez.

**EXE'yi `_internal` klasöründen ayırmayın.** Qt, Python ve AI bileşenleri bu klasördedir. Paket dijital olarak imzalı değildir; Windows yayıncı uyarısı gösterebilir. İndirmenin SHA-256 değerini release içindeki `SHA256SUMS.txt` ile karşılaştırabilirsiniz.

## İlk deneme

1. **Encode / Decode → 1. Files** bölümünden bir görüntü seçin.
2. Yöntemi seçin: **JPEG**, **JPEG2000**, **DCT** veya **DWT**.
3. Kayıplı DCT/DWT için **Target BPP** ya da **Target PSNR** hedeflerinden yalnızca birini seçin. JPEG'de quality, JPEG2000'de rate kullanılır.
4. Encode/decode işlemini çalıştırın. Original, Decoded, Restored, Difference ve Compare görünümlerini inceleyin.
5. Çıktıyı kalıcı tutmak için ilgili **Save** düğmesine basın. İşleme sırasında geçici dosyalar kullanılabilir; kullanıcı çıktıları Save ile kaydedilir. Uygulama tercihleri ayrıca saklanır.

Sunuma başlangıç için 512×512 bir görüntüde DWT/db4/level 3 ve DCT ile 0.2, 0.4, 0.8, 1.2 BPP hedeflerini deneyebilirsiniz. Görsel kalite görüntü içeriğine bağlıdır; hedef ile gerçekleşen değeri aynı şey kabul etmeyin.

## Özellikler

| Alan | İşlev |
| --- | --- |
| Sıkıştırma | Kayıplı DWT, 8×8 blok DCT ve PR-QMF; tersinir 5/3 DWT ile kayıpsız mod; standart JPEG/JPEG2000 |
| Compare | Kaydırmalı ve yan yana karşılaştırma, tam ekran, hata haritaları, histogram, aşamalı görüntüler ve 8-bit gri seviye önizleme |
| Dönüşümler | Qt ile çizilen DWT ağacı, LL/LH/HL/HH bantları, DCT DC/AC açıklamaları ve blok/ızgara görünümleri |
| Wavelet karşılaştırması | DWT karşılaştırma modu seçildiğinde db4, db8 ve db12 sonuçlarını birlikte inceleme |
| PSNR–BPP grafiği | Aynı fotoğrafın DCT/DWT sonuçları: yatay **gerçekleşen BPP**, dikey **PSNR (dB)**; DCT mavi, DWT turuncu noktalar |
| AI, ROI and restoration | Seçilen nesne/bölgelere öncelik verme, isteğe bağlı YOLO analizi ve restorasyon araçları |
| Benchmark | Birden çok yöntem/ayar için ölçümler ve CSV/JSON raporları |
| Video / Transport | Eğitim amaçlı kare/GOP sıkıştırma, paketleme, paket kaybı deneyleri ve UDP aktarımı |

Video/Transport, standart H.264/H.265 kodlayıcısı veya gerçek bir 5G ağ modeli değildir. YOLO için isteğe bağlı bağımlılıklar ve model ağırlıkları gerekir. Eğitilmiş restorasyon modeli yoksa temel iyileştirme yolu kullanılabilir; bunu öğrenilmiş model sonucu olarak yorumlamayın. Gri seviye comparison seçeneği kaynağı değiştirmeyen bir önizlemedir.

### Ölçümleri yorumlama

- **BPP** = kodlanmış dosyanın toplam bit sayısı / (genişlik × yükseklik). Başlık maliyeti dahildir; RGB'de üç kanala ayrıca bölünmez.
- **PSNR**, orijinal ve yeniden oluşturulan görüntü arasındaki MSE'den hesaplanır. 8-bit görüntüde tepe değer 255'tir. Aynı PSNR her görüntüde aynı algısal kaliteyi garanti etmez.
- **SSIM**, global SSIM uygulamasıdır; kayan pencereli başka kütüphanelerin sonuçlarıyla birebir aynı değildir.
- **Compression ratio**, ham piksel belleği / kodlanmış dosya boyutu esasına dayanır. Kaynak dosyanın diskteki boyutu ayrıca gösterilebilir; PNG/JPEG girişi zaten sıkıştırılmış olabilir.
- Çok düşük BPP özellikle renkli veya karmaşık dokulu görüntülerde belirgin kayba neden olabilir. Bazı hedefler dosya başlığı ve codec sınırları nedeniyle ulaşılamayabilir.
- GUI'de Step yerine tek bir kalite/boyut hedefi seçilir; nicemleme adımı içeride aranır. CLI'da `--step` uyumluluk için korunur.

## Kaynak koddan çalıştırma

Windows için doğrulanan ortam: **Python 3.12 x64**, **PySide6 6.8.3**. Qt DLL uyumsuzluklarını önlemek için yeni bir sanal ortam kullanın; farklı Python kurulumlarından DLL kopyalamayın.

```powershell
git clone https://github.com/icliberen/codec-project.git
cd codec-project
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt "PySide6==6.8.3"
.\.venv\Scripts\python.exe start_gui.pyw
```

İsteğe bağlı video ve AI özellikleri:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-video.txt -r requirements-optional.txt
.\.venv\Scripts\python.exe scripts/download_models.py
```

Model komutu resmi Ultralytics sürümünden iki YOLO11 ağırlığı indirir ve SHA-256 doğrular. Ağırlıklar kaynak depoda tutulmaz. JPEG2000, Pillow'un OpenJPEG desteğini gerektirir.

### Komut satırı

Aşağıdaki komutları proje kökünde, sanal ortamınızın Python'u ile çalıştırın:

```powershell
python -m smartcodec generate-samples samples
python -m smartcodec encode input.tiff output.swc --codec dwt --wavelet db4 --level 3 --quantizer scalar --target-bpp 0.4
python -m smartcodec encode input.tiff output-dct.swc --codec dct --target-psnr 30
python -m smartcodec decode output.swc decoded.png
python -m smartcodec encode input.tiff output.jpg --codec jpeg --quality 30
python -m smartcodec encode input.tiff output.jp2 --codec jpeg2000 --rate 20
python -m smartcodec --help
```

Kayıpsız mod yalnızca tersinir DWT yolunu kullanır; bu modda BPP/PSNR hedefi veya ROI/restorasyon uygulanmaz.

## Doğrulama ve Windows paketi

```powershell
.\.venv\Scripts\python.exe -m compileall -q smartcodec scripts start_gui.pyw
.\.venv\Scripts\python.exe scripts/verify_release.py
powershell -ExecutionPolicy Bypass -File .\build_windows.ps1 -PythonExe .\.venv\Scripts\python.exe
powershell -ExecutionPolicy Bypass -File .\validate_windows_package.ps1 -MinimalEnvironment -StartupSeconds 30
```

Derleme komutu sabitlenmiş bağımlılıkları kurar, resmi YOLO ağırlıklarını doğrular, lisans bildirimlerini toplar ve `dist/Smart Codec` klasörünü oluşturur. Mevcut `dist/Smart Codec` build çıktısını günceller. Windows sistem fontları pakete kopyalanmaz.

`scripts/verify_release.py` küçük sentetik görüntülerde codec round-trip, hedeflerin birbirini dışlaması ve Qt tema/dil açılış kontrollerini yapar; kapsamlı kalite değerlendirmesi değildir. Paket doğrulaması EXE'yi sınırlı PATH ile açar, ana pencereyi kontrol eder ve yalnızca başlattığı işlemi kapatır.

[Windows release workflow](.github/workflows/release.yml), ilk yayımlama sırasında veya Actions üzerinden manuel çalıştırıldığında paketi oluşturur. Kontroller başarılıysa ZIP yayımlanır. Sürüm `pyproject.toml` içindedir; mevcut release'in üzerine yazılmaz.

## Kod düzeni

```text
smartcodec/                   Codec, CLI, Qt arayüzü, benchmark, AI ve transport
assets/                       SVG simgeleri
scripts/                      Model indirme, lisans toplama ve release kontrolü
start_gui.pyw                 Qt uygulamasının giriş noktası
build_windows.ps1            Windows paketleme
validate_windows_package.ps1 Paket açılış kontrolü
requirements*.txt            Temel, isteğe bağlı ve build bağımlılıkları
LICENSE                      AGPL-3.0 lisansı
THIRD_PARTY_NOTICES.txt       Üçüncü taraf bildirimleri
```

Eski raporlar, kişisel ayarlar, önbellekler, çıktılar, EXE/runtime dosyaları ve büyük model ağırlıkları Git geçmişine dahil değildir. Yeniden dağıtım hakları doğrulanmamış test fotoğrafları eklenmemiştir; kendi görüntülerinizi veya `generate-samples` çıktısını kullanabilirsiniz.

## Lisans

Copyright © 2026 icliberen. Smart Codec kaynak kodu **GNU AGPL-3.0-only** altında yayımlanır; [LICENSE](LICENSE) tam metni içerir. Yazılım garanti verilmeden sunulur.

Ultralytics kodu ve YOLO ağırlıkları, ticari lisans olmadan [Ultralytics AGPL-3.0 koşullarına](https://www.ultralytics.com/license) tabidir. Diğer bağımlılıklar kendi lisanslarını korur; [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) ve Windows paketindeki lisans dizinine bakın. Windows font dosyaları dağıtılmaz.
