
class SegmentationParams:
    def __init__(self, oraganelle: str):
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
        self.do_remove_small_objects = False
        self.do_fill_holes = False
        self.do_fill_holes_boarders = False
        

        # Configure based on oraganelle
        if oraganelle == 'Membrane':
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
            self.do_remove_small_objects = False
            self.do_fill_holes = False
            self.do_fill_holes_boarders = False

        elif oraganelle == 'ER':
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
            self.do_remove_small_objects = False
            self.do_fill_holes = False
            self.do_fill_holes_boarders = False

        elif oraganelle == 'Mito':
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
            self.do_remove_small_objects = False
            self.do_fill_holes = False
            self.do_fill_holes_boarders = False
            
        elif oraganelle == 'Micro':
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
            self.do_remove_small_objects = False
            self.do_fill_holes = False
            self.do_fill_holes_boarders = False
            
        elif oraganelle == 'NucEnv':
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
            self.do_remove_small_objects = False
            self.do_fill_holes = False
            self.do_fill_holes_boarders = False
            
        elif oraganelle == 'DNA':
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
            self.do_remove_small_objects = False
            self.do_fill_holes = False
            self.do_fill_holes_boarders = False
            
        elif oraganelle == 'AF':
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
            self.do_remove_small_objects = False
            self.do_fill_holes = False
            self.do_fill_holes_boarders = False
            
        elif oraganelle == 'Nucleoli':
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
            self.do_remove_small_objects = False
            self.do_fill_holes = False
            self.do_fill_holes_boarders = False

        else:
            raise ValueError(f"Unknown oraganelle: {oraganelle}")

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


