import lensless.lenslessConverter as lenslessConverter

class LenslessConversion:
    def __init__(self):
        pass

    def __call__(self, pic):
        lensless_image = lenslessConverter.convert_image_to_lensless(pic)
        # lensless_image = lenslessDctTranslation.lensless_and_dct_translation(pic)
        return lensless_image

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"