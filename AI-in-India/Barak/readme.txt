Dataset: https://wiki.cancerimagingarchive.net/plugins/servlet/mobile?contentId=19726546&fbclid=IwAR0h9kvbbmpjLioTjQ8EGJ0Jnr6Ea1VvT5Xge_aOUaXyFPGBbr2NcpjVkMc#content/view/19726546

a) Download the dataset
b) Convert .dicom files of patients to .mha files using file "convert_to_mha.py"
c) Extract .npy matrix of CT volume of each patient using file "convert_to_numpy.py"
d) Candidate Generation (Go to "Segmentation" folder):
	i) Generate Axial training slices with corresponding mask using "test_itk_1.py"
	ii) Train U-Net for candidate generation using file "model11_new.py"
	iii) Results are already present in "/Segmentation/results/" for each test axial slice.
e) False Positive Reduction
	i) For each input representation (eg., 2.5D-1, etc.), there are two files, one for 18x augmentation using file "data_preprocess_*.py" and other for 34x augmentation using file "data_preprocess1_*.py"
	ii) To train 2D CNN with Test Time Augmentation use, file "kfold.py"
	iii) To train 2D CNN without Test Time Augmentation use, file "kfold1.py"
	iv) To train 3D CNN for "3D" category, use file "kfold1_3.py"
	v) To train SVM on last layer of 2D CNN and 3D CNN, use "svm.py"

The presentation is named "AI Barak.pdf"

ROC curves are already present for each category and sub-categories in "/ROC_CURVES"

Results of Candidate generation are present in "/FIGURES"
