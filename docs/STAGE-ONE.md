# Tahap pertama Credit Score

Historical milestone: this report describes its recorded state. The current public input-range behavior is documented in [SERVING-POLICY.md](SERVING-POLICY.md); final delivery is recorded in [RELEASE.md](RELEASE.md).

21 Oktober 2026. Preview lokal berjalan di `http://127.0.0.1:8518` dari folder kerja `CreditScore`. Arsip tugas di SOURCES tetap utuh. Versi ini belum dipush atau dideploy ke layanan publik.

## Hasil

Jalur prediksi kini memakai satu sklearn Pipeline berisi parser, imputasi median, pembatasan outlier yang dipelajari dari train, dan classifier. Pipeline lengkap disimpan bersama manifest. Mode lokal dan handler SageMaker memuat bundle yang sama; tidak ada fit ulang atau penghitungan median dari profil yang sedang diprediksi.

Kontrak 1.0 menetapkan 18 fitur bernama dan tiga kelas Good, Poor, Standard. Fitur hilang, fitur asing, angka tidak terbatas, boolean, teks ambigu, count pecahan dan nilai di luar domain ditolak. Nilai null yang diizinkan diimputasi menggunakan median training dan diberi keterangan pada respons. [Kontrak dan keputusan cleaning](DATA-CONTRACT.md).

Parser umur riwayat kredit menerima teks sumber dan bulan numerik. Parser pembayaran menerima Yes/No dan 0/1. Memproses ulang nilai numerik tidak lagi mengosongkan dua fitur tersebut. Tanda minus pada count tidak dibuang untuk menghasilkan angka positif; nilai count negatif menjadi missing dalam data training. Fitur bertanda yang diizinkan mempertahankan tandanya.

Contoh dan batas input aplikasi berasal dari training, dengan unit uang dataset yang belum teridentifikasi. Default lama dalam puluhan/ratusan juta tidak dipakai. Klaim “sangat aman”, “zona aman” dan arahan pemberian kredit diganti penjelasan kategori dataset. Hasil tersimpan di session state dan dihapus saat input, contoh atau sumber prediksi berubah.

Respons tanpa label, label asing, daftar label kosong, versi tidak cocok dan kegagalan layanan menjadi error teknis, bukan kategori Poor. Mode AWS tidak berpindah diam-diam ke model lokal. Endpoint harus memakai versi bundle yang sesuai dengan konfigurasi input.

## Bundle integrasi

- Model: `stage1-b098e3ec6e0b`.
- SHA-256 model: `88e870bb1b42aea68be69909f1f8beb9a724d0d56b81977949599df9c63ad25c`.
- Sumber CSV: `464a762d352a33ce6f18ae42a1bc729d8da5aa03e50d8736e0ef83d854c22412`.
- Training: 19.989 baris / 9.003 nasabah. Holdout tersimpan: 5.011 baris / 2.251 nasabah. Tidak ada Customer_ID beririsan.
- Model awal: Random Forest 100 estimator, seed 42, dua jobs. Tidak ada tuning atau pemilihan model berdasarkan holdout.
- Python 3.12, sklearn 1.9.1, NumPy 2.2.6, pandas 2.2.3, joblib 1.6.0; dependency lainnya dikunci dalam uv.lock.

Model dibangun ulang agar preprocessing dapat diserialisasi dengan benar. Ini adalah build untuk menguji integrasi, bukan hasil evaluasi final. Angka accuracy historis 73,24%/73,56% tidak dipindahkan ke model baru. Holdout belum dipakai untuk melaporkan performa. Metadata dan split indices pribadi disimpan agar tahap kedua dapat melanjutkan protokol yang sama.

## Pemeriksaan

| Pemeriksaan | Hasil dan cakupan |
|---|---|
| Ruff | Lulus untuk paket Python, UI, handler, scripts dan tests |
| Pytest | 74 tes lulus, termasuk state Streamlit dengan bundle lokal nyata |
| Paritas prediksi | 100 profil sintetis dalam domain: Pipeline.predict langsung, service lokal dan round trip JSON handler SageMaker identik |
| Parser | Format sumber/numerik, idempotensi, tanda minus, malformed text, missing values dan statistik train yang tidak berubah saat inference |
| Error | Skema salah, unknown/missing feature, NaN/Infinity, out-of-domain, response kosong/asing, timeout, bundle hilang/rusak dan versi tidak cocok |
| Paket SageMaker | Paket diekstrak dan dijalankan dalam proses Python baru; hasil sama dengan lokal; CSV dan split indices tidak disertakan |
| Browser | Chrome 155 pada 1440×900, 390×900, 320×900: prediksi nyata, edit menghapus hasil, reset, endpoint belum diatur dan pemulihan lokal; tanpa page error/overflow |
| Arsip | Hash 20 file sumber yang tercatat pada audit tetap sama |

Kontrak UI dengan respons cloud buatan dipakai untuk memastikan bug lama tidak muncul kembali; itu bukan panggilan endpoint AWS. [Pemeriksaan paket](verification/sagemaker-package.json) dan [hasil browser](verification/browser-stage-one.json) mencatat batas verifikasi.

Pemeriksaan visual menemukan caption dan tombol utama berkontras rendah, serta urutan heading yang melompat. Keterangan dipindahkan ke teks normal, warna utama diperdalam, dan heading diperbaiki. Pemeriksaan final tidak lagi menemukan color-contrast atau heading-order. Temuan axe yang masih terkait markup widget Streamlit dicatat dalam hasil browser; tahap pertama ini tidak mengklaim audit aksesibilitas penuh selesai. Khususnya, tombol stepper angka bawaan belum memiliki nama aksesibel pada runtime Streamlit 1.55. Perbaikan/alternatif widget dimasukkan ke pekerjaan UX berikutnya.

## Melanjutkan

Tahap kedua: dokumentasikan provenance dan unit sumber; ukur baseline dan model pembanding pada validasi yang memisahkan nasabah; lakukan analisis sensitivitas cleaning dan kalibrasi; laporkan performa per kelas beserta ketidakpastian. Holdout final harus tetap dipisahkan dari tuning. UI dan dokumentasi CV hanya memakai metrik baru setelah run evaluasi selesai.

Paket cloud sudah disiapkan untuk kontrak baru, tetapi deployment dan kesehatan endpoint AWS nyata belum diperiksa. Container lama sklearn 1.4-2 tidak sesuai versi bundle ini. Penyesuaian container dan rilis cloud menjadi langkah terpisah dari verifikasi lokal tahap pertama.
