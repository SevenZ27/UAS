# Credit Score: UI, hasil prediksi, dan aksesibilitas

Historical milestone: this report describes its recorded state. The current public input-range behavior is documented in [SERVING-POLICY.md](SERVING-POLICY.md); final delivery is recorded in [RELEASE.md](RELEASE.md).

Tahap ketiga memperbaiki aplikasi **Streamlit 1.55**, tanpa mengganti model `stage2-49c6876730d2` atau menilai ulang holdout untuk memilih model. Hasil prediksi kini memiliki panel kategori tersendiri, ringkasan input yang dipakai, dan konteks benchmark yang dapat dibuka saat diperlukan.

## Formulir dan alur hasil

Formulir tetap memuat 18 fitur dalam tiga kelompok. Desktop menempatkan formulir dan panel hasil berdampingan. Di bawah 1000 px keduanya bertumpuk; layar kecil menampilkan satu input per baris. Pasangan field dirender per baris sehingga urutan baca dan keyboard tetap konsisten dengan susunan visual.

Tujuh belas input angka kini memakai field teks dengan petunjuk keyboard numerik dan validasi eksplisit. Widget angka sebelumnya dapat mengembalikan input kosong ke nilai minimum, serta menahan draft di luar rentang meskipun tombol reset sudah ditekan. Penggantian kontrol menghilangkan perilaku tersebut sekaligus menghapus tombol stepper tanpa nama, tanpa menyembunyikan kontrol dari pemeriksaan aksesibilitas.

Titik dan koma desimal diterima. Format dengan campuran pemisah ribuan/desimal, teks, NaN, Infinity dan angka yang melampaui representasi finite ditolak. Pecahan pada fitur hitungan diteruskan ke validator kontrak untuk ditolak, bukan dipotong menjadi bilangan bulat. Domain model diperiksa sebelum inference lokal maupun pengiriman AWS; nilai di luar rentang tidak diam-diam diganti. Pesan error menyebut fitur yang perlu diperbaiki dan mempertahankan isian lain.

Usia wajib diisi. Fitur opsional yang kosong tetap menjadi null dan memakai median training asli di pipeline; hasil menampilkan peringatan imputasi. Pilihan pembayaran minimum memiliki keadaan Belum diketahui. Contoh awal menggabungkan median training, dibulatkan dua desimal untuk fitur nonhitungan dan dijelaskan pada halaman. Nilai yang terlihat itulah yang dikirim sebagai input; pembulatan contoh tidak mengubah median imputasi atau binary model.

Setelah prediksi berhasil, fokus berpindah ke judul hasil satu kali. Error memindahkan fokus ke ringkasan masalah. Mengedit input tidak memindahkan fokus dan menghapus hasil lama setelah perubahan dikomit dengan Enter atau meninggalkan field. Reset contoh dan perubahan backend juga menghapus hasil lama. Gangguan teknis tidak pernah menjadi kategori Poor.

## Visualisasi dan keterlacakan

Daftar Good, Standard dan Poor menandai kategori terpilih dengan teks serta garis tepi. Tidak ada gauge angka, skor kelayakan, probabilitas gagal bayar atau penjelasan sebab-akibat individual yang belum divalidasi. Keadaan awal tidak memilih salah satu kategori.

Input saat prediksi dan versi model tersedia dalam disclosure. Unduhan JSON berisi profil numerik yang benar-benar dipakai, respons model, peringatan dan batas makna hasil. Tidak ada identitas nasabah atau baris CSV dalam file tersebut.

Disclosure benchmark menampilkan perbandingan accuracy model/baseline pada skala nol sampai 100%, interval bootstrap, macro-F1 dan cakupan kontrak. Confusion matrix memakai header baris/kolom, jumlah observasi dan persentase per kelas aktual. Tabel muat pada layar 320 px tanpa horizontal scroll; label dan angka tetap tersedia sebagai teks semantik.

Semua angka benchmark dibaca dari laporan tahap kedua setelah hash laporan, versi model, hash binary dan urutan kelas cocok dengan metadata aktif. Jika laporan tidak cocok, angka evaluasi disembunyikan dengan pesan yang jelas; prediksi lokal tetap berjalan. Precision Good, kelemahan recall Poor pada usia 45+, cakupan 61,2%, serta batas validasi eksternal tetap dinyatakan.

## Aksesibilitas dan bukti uji

Sebanyak **110 tes lulus**, mencakup kontrak, model, split, state UI, parsing angka, input kosong, error domain, ketidaksesuaian laporan dan kegagalan backend. [JUnit](verification/tests-stage-three.xml) dan [manifest verifikasi](verification/stage-three.json) menyimpan hasilnya.

Playwright CLI dengan Chrome memeriksa lebar 1440, 900, 390 dan 320 px. Dua ukuran mobile memakai touch emulation dan tap pada aksi prediksi. Alur yang diperiksa meliputi keadaan awal, hasil, edit, reset, usia kosong, fitur opsional kosong, penolakan nilai di luar rentang, kegagalan konfigurasi AWS, pemulihan mode Lokal, unduhan dan disclosure evaluasi. Tidak ditemukan horizontal page overflow atau JavaScript page error. Tes browser tidak memanggil endpoint AWS nyata.

Axe memeriksa keadaan dengan pengaturan backend, ringkasan input dan benchmark terbuka: **nol pelanggaran terdeteksi pada keempat ukuran**. Tiga ukuran menyisakan satu pemeriksaan kontras yang tidak dapat dihitung otomatis karena paragraf sebagian tertutup toolbar Streamlit saat posisi scroll tertentu. Pemeriksaan manual mengukur teks rgb(49,51,63) pada putih dengan rasio **12,53:1**. Temuan incomplete tetap disimpan, tidak dihapus atau dikecualikan dari hasil axe.

Pemeriksaan keyboard terpisah memastikan skip link bisa dicapai, fokus terlihat dengan outline 3 px, urutan field sesuai visual, dan prediksi dapat dijalankan dengan Enter. Reflow dengan root text size 200% pada viewport 720 px tidak menghasilkan horizontal page overflow. Reduced motion memakai perpindahan fokus dengan scroll langsung dan menghilangkan transisi warna opsional. Ini adalah simulasi browser; belum pengujian perangkat iPhone fisik, Safari atau pembaca layar oleh pengguna, dan bukan sertifikasi WCAG.

Adapter JavaScript lokal yang kecil melengkapi landmark main, bahasa dokumen Indonesia, inputmode dan required-age setelah rerender Streamlit. Tidak ada library eksternal yang dimuat oleh adapter. Selector perlu diperiksa kembali saat versi framework ditingkatkan. Lihat [catatan desain](DESIGN.md).

## Batas verifikasi dan kelanjutan

Hash binary model, sumber CSV, indeks split dan laporan evaluasi tahap kedua tetap cocok dengan bukti sebelumnya. Dua puluh file arsip tugas asli juga tetap sama. Perubahan ini tersedia di aplikasi lokal; tidak menyatakan deployment AWS atau publik telah diperbarui.

Dokumentasi portofolio, CV dan publikasi dapat memakai laporan tahap kedua untuk klaim model serta laporan ini untuk klaim implementasi UI. Angka benchmark tetap merupakan klasifikasi label dataset pada nasabah terpisah, belum pembuktian kelayakan kredit dunia nyata.
