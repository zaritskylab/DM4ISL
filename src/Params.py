def training_seeds(organelle):
    if organelle == 'DNA':
        train_test_split_seed = 35 
        test_patches_seed = 5
    if organelle == 'Nuclear envelope':
        train_test_split_seed = 4 
        test_patches_seed = 5
    if organelle == 'Nucleoli':
        train_test_split_seed = 4 
        test_patches_seed = 5
    if organelle == 'Actin filament':
        train_test_split_seed = 20 
        test_patches_seed = 5
    if organelle == 'Microtubules':
        train_test_split_seed = 3 
        test_patches_seed = 16
    if organelle == 'Mitochondria':
        train_test_split_seed = 3 
        test_patches_seed = 16
    return train_test_split_seed, test_patches_seed


def uncertaintymap_parameters(organelle):
    if organelle == 'DNA':
        std_th = 0.045
    if organelle == 'Nuclear envelope':
        std_th = 0.055
    if organelle == 'Nucleoli':
        std_th = 0.065
    if organelle == 'Actin filament':
        std_th = 0.09
    if organelle == 'Microtubules':
        std_th = 0.04
    return std_th

def organelle_parameters(organelle):            
    if organelle == 'DNA':
        std_TH = 20
        BFmaxGL_clip = 55000
        BFminGL_clip = 10000
        FLmaxGL_clip = 650
        FLminGL_clip = 380
    if organelle == 'Mitochondria':
        std_TH = 18 
        BFmaxGL_clip = 55000
        BFminGL_clip = 10000
        FLmaxGL_clip = 800
        FLminGL_clip = 385
    if organelle == 'Nuclear envelope':
        std_TH = 70 
        BFmaxGL_clip = 55000
        BFminGL_clip = 12000
        FLmaxGL_clip = 1500
        FLminGL_clip = 390
    if organelle == 'Nucleoli':
        std_TH = 53
        BFmaxGL_clip = 55000
        BFminGL_clip = 10000
        FLmaxGL_clip = 2500
        FLminGL_clip = 390
    if organelle == 'Actin filament':
        std_TH = 25
        BFmaxGL_clip = 35000
        BFminGL_clip = 12000
        FLmaxGL_clip = 800
        FLminGL_clip = 390
    if organelle == 'Microtubules':
        std_TH = 250
        BFmaxGL_clip = 55000
        BFminGL_clip = 10000
        FLmaxGL_clip = 5000
        FLminGL_clip = 395
    return std_TH, BFmaxGL_clip, BFminGL_clip, FLmaxGL_clip, FLminGL_clip


class SegmentationParams:
    def __init__(self, organelle: str):
        # Default parameters
        self.remove = []
        self.organelle_th = -2 #### -2 - single otsu th for 3D patch , -1 - otsu th for every slice in 3D patch , gl 0-255 maual th for 3D patch
        self.filter_type = 'median' # 'bilateral' 'median's
        self.filter_kernel = 3
        self.sigma = 3
        self.do_morph = False
        self.k1 = 3
        self.k2 = 3
        self.k3 = 3
        self.do_erode_dilate = False
        self.do_remove_small_objects = True
        self.do_fill_holes = True
        self.do_fill_holes_boarders = False

        if organelle == 'Mitochondria':
            self.remove = []
            self.organelle_th = -2
            self.filter_type = 'median' # 'bilateral' 'median's
            self.filter_kernel = 3
            self.sigma = 3
            self.do_morph = False
            self.k1 = 3
            self.k2 = 3
            self.k3 = 3
            self.do_erode_dilate = False
            self.do_remove_small_objects = True
            self.do_fill_holes = True
            self.do_fill_holes_boarders = False
            
        elif organelle == 'Microtubules':
            self.remove = []
            self.organelle_th = -2
            self.filter_type = 'median' # 'bilateral' 'median's
            self.filter_kernel = 3
            self.sigma = 15
            self.do_morph = False
            self.k1 = 3
            self.k2 = 3
            self.k3 = 3
            self.do_erode_dilate = False
            self.do_remove_small_objects = True
            self.do_fill_holes = True
            self.do_fill_holes_boarders = False
            
        elif organelle == 'Nuclear envelope':
            self.remove = []
            self.organelle_th = -2
            self.filter_type = 'median' # 'bilateral' 'median's
            self.filter_kernel = 3
            self.sigma = 70
            self.do_morph = False
            self.k1 = 3
            self.k2 = 3
            self.k3 = 3
            self.do_erode_dilate = False
            self.do_remove_small_objects = True
            self.do_fill_holes = True
            self.do_fill_holes_boarders = False
            
        elif organelle == 'DNA':
            self.remove = []
            self.organelle_th = -2
            self.filter_type = 'median' # 'bilateral' 'median's
            self.filter_kernel = 5
            self.sigma = 50
            self.do_morph = False
            self.k1 = 3
            self.k2 = 3
            self.k3 = 3
            self.do_erode_dilate = False
            self.do_remove_small_objects = True
            self.do_fill_holes = True
            self.do_fill_holes_boarders = False
            
        elif organelle == 'Actin filament':
            self.remove = []
            self.organelle_th = -2
            self.filter_type = 'median' # 'bilateral' 'median's
            self.filter_kernel = 3
            self.sigma = 15
            self.do_morph = False
            self.k1 = 3
            self.k2 = 3
            self.k3 = 3
            self.do_erode_dilate = False
            self.do_remove_small_objects = True
            self.do_fill_holes = True
            self.do_fill_holes_boarders = False
            
        elif organelle == 'Nucleoli':
            self.remove = []
            self.organelle_th = -2
            self.filter_type = 'median' # 'bilateral' 'median's
            self.filter_kernel = 3
            self.sigma = 50
            self.do_morph = True
            self.k1 = 3
            self.k2 = 3
            self.k3 = 3
            self.do_erode_dilate = False
            self.do_remove_small_objects = True
            self.do_fill_holes = True
            self.do_fill_holes_boarders = False

        else:
            raise ValueError(f"Unknown organelle: {organelle}")

    def __repr__(self):
        return (
                f"SegmentationParams(remove={self.remove}, "
                f"filter_type={self.filter_type}, "
                f"organelle_th={self.organelle_th}, "
                f"filter_kernel={self.filter_kernel}, "
                f"sigma={self.sigma}, "
                f"do_morph={self.do_morph}, "
                f"k1={self.k1}, "
                f"k2={self.k2}, "
                f"k3={self.k3}, "
                f"do_erode_dilate={self.do_erode_dilate}, "
                f"do_remove_small_objects={self.do_remove_small_objects}, "
                f"do_fill_holes={self.do_fill_holes}, "
                f"do_fill_holes_boarders={self.do_fill_holes_boarders},) "
                
               )


