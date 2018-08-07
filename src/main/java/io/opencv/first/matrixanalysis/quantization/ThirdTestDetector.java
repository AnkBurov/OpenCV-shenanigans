package io.opencv.first.matrixanalysis.quantization;

import io.opencv.OpenCvBased;
import io.opencv.util.Matrices;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ThreadLocalRandom;

import static io.opencv.first.matrixanalysis.SamePixelColorMatricsFinder.cropImage;
import static org.opencv.imgproc.Imgproc.CHAIN_APPROX_SIMPLE;
import static org.opencv.imgproc.Imgproc.MORPH_OPEN;
import static org.opencv.imgproc.Imgproc.RETR_TREE;

/**
 * Works well if vin number is white and thick and everything else is not white
 */
public class ThirdTestDetector extends OpenCvBased {
    public static void main(String[] args) throws Exception {
        File file = new ClassPathResource("images/vin4.jpeg").getFile();

        try (Matrices matrices = new Matrices()) {
            Mat origPic = matrices.fromSupplier("orig_pic", () -> Imgcodecs.imread(file.getAbsolutePath()));

            // crop image
            Mat cropped = cropImage(matrices, origPic, 0.85);

            // get grayscale
            Mat gray = matrices.fromSupplier("gray", Mat::new);
            Imgproc.cvtColor(cropped, gray, Imgproc.COLOR_RGB2GRAY);
            Imgcodecs.imwrite(file.getName() + "_gray.jpg", gray);

            // dilate
            Mat dilate = matrices.fromSupplier("dilate", gray::clone);
            Mat morphKernel = matrices.fromSupplier("morph_kernel", () -> Imgproc.getStructuringElement(MORPH_OPEN, new Size(45, 45)));
            Imgproc.dilate(dilate, dilate, morphKernel);
            Imgcodecs.imwrite(file.getName() + "_dilate.jpg", dilate);

            // erode
            Mat erode = matrices.fromSupplier("erode", dilate::clone);
            Mat erodeKernel = matrices.fromSupplier("morph_kernel2", () -> Imgproc.getStructuringElement(MORPH_OPEN, new Size(45, 45))); //21 21 maybe
            Imgproc.erode(erode, erode, erodeKernel);
            Imgcodecs.imwrite(file.getName() + "_erode.jpg", erode);

            // blur image
            Mat blur = matrices.fromSupplier("blur", Mat::new);
            Imgproc.GaussianBlur(erode, blur, new Size(5, 5), 0);
            Imgcodecs.imwrite(file.getName() + "_blur.jpg", blur);

            // canny edge
            Mat canny = matrices.fromSupplier("canny", Mat::new);
            Imgproc.Canny(blur, canny, 15, 255);
            Imgcodecs.imwrite(file.getName() + "_canny.jpg", canny);

            // find contours
            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = matrices.newMatrix("hierarchy");
            Imgproc.findContours(canny, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

            //draw rectangles
            Mat secondRects = matrices.fromSupplier("second_rects", gray::clone);
            contours.stream()
                    .map(Imgproc::boundingRect)
                    .filter(rect -> isBigEnough(rect, cropped))
                    .peek(rect -> {
                        Mat subImage = matrices.fromSupplier(() -> gray.submat(rect));
                        Imgcodecs.imwrite(file.getName() + "_subimage" + ThreadLocalRandom.current().nextInt(1000) +".jpg", subImage);
                    })
                    .forEach(rect -> {
                        Imgproc.rectangle(secondRects, rect.br(), rect.tl(), new Scalar(500));
                    });

            Imgcodecs.imwrite(file.getName() + "_second_rects.jpg", secondRects);
        }
    }

    private static boolean isBigEnough(Rect rect, Mat image) {
        return rect.width > image.cols() / 100 * 30 || rect.height > image.rows() / 100 * 30;
    }
}

/*// and again
            Imgproc.dilate(erode, erode, morphKernel);
            Imgcodecs.imwrite(file.getName() + "_dilate_again.jpg", erode);

            Imgproc.erode(erode, erode, erodeKernel);
            Imgcodecs.imwrite(file.getName() + "_erode_again.jpg", erode);
*/