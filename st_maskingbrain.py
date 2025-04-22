import streamlit as st
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.ndimage as ndi
import tempfile
import os
from pathlib import Path
from skimage import exposure, filters, morphology, measure, util
from skimage import measure
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
from skimage.exposure import equalize_adapthist
from scipy.ndimage import median_filter, gaussian_filter
from matplotlib.colors import ListedColormap
import glob
import math
from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
import cv2

st.set_page_config(page_title="MRI Corpus Callosum Analyzer", layout="wide")
st.title("🧠 MRI Corpus Callosum Segmentation and Analysis")
st.header("Final Project ETS - DEMO 23 April 2025")
st.subheader("By Nurul Annisa - 5023221031")

st.sidebar.header("Step 1: Upload Folder Berisi DICOM Files")

dicom_dir = st.sidebar.text_input("Path ke folder DICOM", value=r"D:\YA ALLAH SEMESTER 6\04 - PCM\VS CODE\FP MASKING BRAIN\SE000001")

def normalize_display(img):
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min())


data_slices = []
if dicom_dir:
    try:
        all_files = list(Path(dicom_dir).glob("*"))
        if not all_files:
            st.warning("Folder tidak mengandung file DICOM.")
        else:
            for path in all_files:
                try:
                    ds = pydicom.dcmread(str(path))
                    data_slices.append(ds)
                except Exception as e:
                    st.warning(f"Gagal membaca {path.name}: {e}")

            try:
                data_slices = sorted(data_slices, key=lambda x: x.SliceLocation)
            except:
                st.warning("Tidak semua DICOM memiliki atribut SliceLocation. Urutan asli digunakan.")

            images = np.stack([s.pixel_array for s in data_slices])
            st.sidebar.success(f"{len(images)} slices berhasil dimuat dengan ukuran {images.shape[1:]}.")

            st.subheader("#Step 1 - Deskripsi dan Upload Citra")
            slice_idx = st.slider("Pilih Slice", 0, len(images)-1, len(images)//2)
            image_slice = images[slice_idx]
            
            # Tambahan: crop dari baris ke-40 ke bawah
            image_slice = image_slice[40:, :]

            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("""
                | Attribute      | Value |
                |----------------|-------|
                | Dtype          | {}    |
                | Shape          | {} x {} |
                | Min Value      | {:.4f} |
                | Max Value      | {:.4f} |
                """.format(image_slice.dtype, image_slice.shape[0], image_slice.shape[1], image_slice.min(), image_slice.max()))
            with col2:
                st.image(normalize_display(image_slice), caption=f"Slice ke-{slice_idx} (normalized)", clamp=True, use_container_width=True)

            st.header("Step 2: Preprocessing - Filter & AHE")
            col1, col2 = st.columns(2)
            with col1:
                filter_choice = st.selectbox("Pilih Jenis Filter", ("Median", "Gaussian", "Bilateral"))
                # median_size = st.slider("Ukuran Median Filter", 1, 15, 3, step=2)
            with col2:
                ahe_clip = st.slider("AHE Clip Limit", 0.001, 0.01, 0.003, step=0.001)

            if filter_choice == "Median":
                filter_param = st.slider("Ukuran Median Filter", 1, 15, 3, step=2)
                img_filtered = median_filter(image_slice, size=filter_param)
            elif filter_choice == "Gaussian":
                filter_param = st.slider("Sigma Gaussian", 0.1, 5.0, 1.0, step=0.1)
                img_filtered = ndi.gaussian_filter(image_slice, sigma=filter_param)
            elif filter_choice == "Bilateral":
                filter_param = st.slider("Sigma Bilateral", 1, 10, 3, step=1)
                img_filtered = cv2.bilateralfilter(image_slice, d=5, sigmaColor=filter_param, sigmaSpace=filter_param)  # atau pakai bilateral dari OpenCV jika ada

            # img_median = median_filter(image_slice, size=median_size)
            img_filter_norm = normalize_display(img_filtered)
            img_ahe = equalize_adapthist(img_filter_norm, clip_limit=ahe_clip)
            
            # Normalisasi ke [0,1] dulu
            img_ref_norm = normalize_display(image_slice)
            img_enhanced_norm = normalize_display(img_ahe)

            mse_val = mean_squared_error(img_ref_norm, img_enhanced_norm)
            psnr_val = peak_signal_noise_ratio(img_ref_norm, img_enhanced_norm, data_range=1.0)

            st.subheader("Plotting Result - Filter & AHE")
            col3, col4, col10 = st.columns(3)
            with col3:
                st.image(normalize_display(image_slice), caption="Original (normalized)", clamp=True)
                fig1, ax1 = plt.subplots()
                ax1.hist(image_slice.ravel(), bins=50)
                ax1.set_title("Histogram Original")
                st.pyplot(fig1)
            with col4:
                st.image(normalize_display(img_filtered), caption="after Filtering", clamp=True)
                fig2, ax2 = plt.subplots()
                ax2.hist(img_ahe.ravel(), bins=50)
                ax2.set_title("Histogram Filter {filter_choice}")
                st.pyplot(fig2)
            with col10:
                st.image(normalize_display(img_ahe), caption="After AHE", clamp=True)
                fig10, ax10 = plt.subplots()
                ax10.hist(img_ahe.ravel(), bins=50)
                ax10.set_title("Histogram {filter_choice} +ahe")
                st.pyplot(fig10)
            st.markdown(f"**MSE:** {mse_val:.5f} | **PSNR:** {psnr_val:.2f} dB")
           
           
            st.header("Step 2.5: Otsu Thresholding Otomatis")
            # Hitung nilai threshold dari citra enhanced (img_clahe)
            otsu_val = filters.threshold_otsu(img_ahe)
            st.write(f"**Nilai threshold Otsu yang terdeteksi:** {otsu_val:.4f}")
            
            st.header("Step 3: Thresholding & Filtering")
            threshold_val = st.slider("Nilai Threshold", float(img_ahe.min()), float(img_ahe.max()), 0.5, step=0.01)
            corpus = img_ahe >= threshold_val
            
            # Tambahkan analisis area tiap objek dari thresholding awal
            st.subheader("📊 Tabel Area Tiap Objek (Thresholding Awal)")
            label_image = measure.label(corpus)
            regions = measure.regionprops(label_image)
            areas = [region.area for region in regions]
            area_df = pd.DataFrame({
                'Objek ke-': list(range(1, len(areas)+1)),
                'Area': areas
            })

            # min_area = st.slider("Minimum Area", 100, 5000, 1000, step=100)
            min_area = st.number_input("Minimum Area", min_value=0, max_value=5000, value=1000, step=100)
            filtered_img = morphology.remove_small_objects(corpus.astype(bool), min_size=min_area)
            
            max_area = area_df['Area'].max()
            area_df['Keterangan'] = area_df['Area'].apply(lambda x: 'TERBESAR' if x == max_area else '')
            st.dataframe(area_df, height=200)
            
            
            st.subheader("Perbandingan Citra thresholding & remove small objects")
            col5, col6 = st.columns(2)
            with col5:
                st.image(normalize_display(corpus), caption="citra thresholding {threshold_val:.2f}", clamp=True)
                # fig5, ax5 = plt.subplots()
                # st.pyplot(fig5)
            with col6:
                st.image(normalize_display(filtered_img), caption="Citra after removing small objects", clamp=True)
                # fig6, ax6 = plt.subplots()
                # st.pyplot(fig6)
            
            # st.subheader("Otsu Thresholding Kedua")
            # otsu_val2 = filters.threshold_otsu(corpus)
            # st.write(f"**Nilai threshold Kedua Otsu yang terdeteksi:** {otsu_val2:.4f}")

            st.header("Step 4: Masking")
            corpus = filtered_img.astype(float)
            mask_corpus = (corpus >= 0.5) * 1
            st.subheader("Matriks Masking Corpus")
            st.write(mask_corpus)

            img_overlay1 = np.where(mask_corpus, image_slice, 0)
            img_overlay2 = np.where(mask_corpus, img_ahe, 0)

            st.subheader("Overlay Masking dengan Citra")
            fig_overlay, axes = plt.subplots(1, 2, figsize=(8, 8))
            axes[0].imshow(img_overlay1, cmap='gray')
            axes[0].set_title("Original + Masking")
            axes[1].imshow(img_overlay2, cmap='gray')
            axes[1].set_title("Enhanced and filter + Masking")
            st.pyplot(fig_overlay)
            
            st.header("Step 5: Labeling & Feature Extraction")
            labels, nlabels = ndi.label(corpus)
            label_arrays = []
            st.write(f"**There are {nlabels} separate components / objects detected.**")
            
            rand_cmap = ListedColormap(np.random.rand(256,3))
            labels_for_display = np.where(labels > 0, labels, np.nan)
            col7, col8, col9 = st.columns(3)
            with col8:
                fig8, ax8 = plt.subplots(figsize=(8, 8))
                ax8.imshow(image_slice, cmap='gray')
                ax8.imshow(labels_for_display, cmap=rand_cmap, alpha=0.6)
                st.pyplot(fig8)
                
            #labeling corpus untuk region props
            image = corpus
            label_img = label(image)
            regions = regionprops(label_img)
            st.write(label_img) 
            
            st.subheader("Visualisasi Orientasi dan Bounding Box Objek")
            # Buat dua subplot (satu baris dua kolom)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # 1. Visualisasi khusus Corpus Callosum di subplot kiri
            masked_corpus = np.zeros_like(image_slice)
            corpus_region = max(regions, key=lambda r: r.area)

            for coord in corpus_region.coords:
                masked_corpus[coord[0], coord[1]] = 255

            # Tampilkan hasil masking
            ax1.imshow(masked_corpus, cmap='gray')
            y0, x0 = corpus_region.centroid
            orientation = corpus_region.orientation
            x1 = x0 + math.cos(orientation) * 0.5 * corpus_region.minor_axis_length
            y1 = y0 - math.sin(orientation) * 0.5 * corpus_region.minor_axis_length
            x2 = x0 - math.sin(orientation) * 0.5 * corpus_region.major_axis_length
            y2 = y0 - math.cos(orientation) * 0.5 * corpus_region.major_axis_length

            ax1.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
            ax1.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
            ax1.plot(x0, y0, '.g', markersize=15)

            minr, minc, maxr, maxc = corpus_region.bbox
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            ax1.plot(bx, by, '-b', linewidth=2.5)

            ax1.set_xlim(0, image_slice.shape[1])
            ax1.set_ylim(image_slice.shape[0], 0)
            ax1.set_title("Corpus Callosum Masking Bounding Box")
            ax1.axis('off')

            # 2. Visualisasi semua objek di subplot kanan
            ax2.imshow(image_slice, cmap='gray')
            for region in regions:
                y0, x0 = region.centroid
                orientation = region.orientation
                x1 = x0 + math.cos(orientation) * 0.5 * region.minor_axis_length
                y1 = y0 - math.sin(orientation) * 0.5 * region.minor_axis_length
                x2 = x0 - math.sin(orientation) * 0.5 * region.major_axis_length
                y2 = y0 - math.cos(orientation) * 0.5 * region.major_axis_length

                ax2.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
                ax2.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
                ax2.plot(x0, y0, '.g', markersize=15)

                minr, minc, maxr, maxc = region.bbox
                bx = (minc, maxc, maxc, minc, minc)
                by = (minr, minr, maxr, maxr, minr)
                ax2.plot(bx, by, '-b', linewidth=2.5)

            ax2.set_xlim(0, image_slice.shape[1])
            ax2.set_ylim(image_slice.shape[0], 0)
            ax2.set_title("Semua Objek")
            ax2.axis('off')

            st.pyplot(fig)  
            
            #---------ekstraksi fitur-----------------
            props = regionprops_table(
                label_img,
                intensity_image=image_slice,  # <--- ini penting!
                properties=(
            'label',
            'area',
            'bbox_area',
            'centroid',
            'eccentricity',
            'equivalent_diameter',
            'orientation',
            'major_axis_length',
            'minor_axis_length',
            'perimeter',
            'solidity',
            'mean_intensity',
            'max_intensity',
            'min_intensity',
            'weighted_centroid',
            'weighted_moments_hu'
                )
            )

            df = pd.DataFrame(props)
            st.dataframe(df)
            st.download_button("Download Fitur ke Excel", df.to_csv(index=False).encode(), "extractfeatures.csv", "text/csv")
            
            # Deskripsi setiap properti
            property_descriptions = {
                "label": "Label unik untuk tiap objek.",
                "area": "Jumlah piksel dalam objek.",
                "bbox_area": "Luas area kotak pembatas (bounding box).",
                "centroid": "Titik tengah (x, y) dari objek.",
                "eccentricity": "Bentuk elipsitas, dari 0 (lingkaran) ke 1 (garis lurus).",
                "equivalent_diameter": "Diameter lingkaran dengan area yang sama.",
                "orientation": "Arah rotasi utama objek dalam radian.",
                "major_axis_length": "Panjang sumbu utama elips yang cocok dengan objek.",
                "minor_axis_length": "Panjang sumbu minor elips.",
                "perimeter": "Panjang keliling objek.",
                "solidity": "Area objek dibagi area convex hull.",
                "mean_intensity": "Rata-rata intensitas dalam objek.",
                "max_intensity": "Intensitas maksimum dalam objek.",
                "min_intensity": "Intensitas minimum dalam objek.",
                "weighted_centroid": "Centroid berdasarkan intensitas piksel.",
                "weighted_moments_hu": "Moments Hu berbobot intensitas, digunakan untuk analisis bentuk objek yang tidak bergantung pada skala, rotasi, dan translasi."
            }

            # Tampilkan semua deskripsi
            desc_df = pd.DataFrame(
                list(property_descriptions.items()),
                columns=["Properti", "Deskripsi"]
            )

            # Tampilkan sebagai tabel
            st.markdown("## 📌 Deskripsi Lengkap Properti Regionprops")
            st.table(desc_df)  # Atau st.dataframe(desc_df) jika ingin bisa scroll/sort


    except Exception as e:
        st.error(f"Gagal memuat DICOM: {e}")
