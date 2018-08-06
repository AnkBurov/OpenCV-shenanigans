package io.opencv.first.matrixanalysis;

import io.opencv.OpenCvBased;
import io.opencv.util.Matrices;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import static io.opencv.first.matrixanalysis.SamePixelColorMatricsFinder.cropImage;
import static org.opencv.imgproc.Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C;
import static org.opencv.imgproc.Imgproc.CHAIN_APPROX_NONE;
import static org.opencv.imgproc.Imgproc.CHAIN_APPROX_SIMPLE;
import static org.opencv.imgproc.Imgproc.MORPH_OPEN;
import static org.opencv.imgproc.Imgproc.RETR_TREE;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY_INV;

public class AnotherTestDetector extends OpenCvBased {

    public static void main(String[] args) throws Exception {
        File file = new ClassPathResource("images/vin6.jpeg").getFile();

        try (Matrices matrices = new Matrices()) {

            Mat origPic = matrices.fromSupplier("orig_pic", () -> Imgcodecs.imread(file.getAbsolutePath()));

            // crop image
            Mat cropped = cropImage(matrices, origPic, 0.85);

            double widthSomePercent = cropped.cols() / 100 * 1; // 5 is percent
            double heightSomePercent = cropped.rows() / 100 * 1; // 5 is percent

            // get grayscale
            Mat gray = matrices.fromSupplier("gray", Mat::new);
            Imgproc.cvtColor(cropped, gray, Imgproc.COLOR_RGB2GRAY);
            Imgcodecs.imwrite(file.getName() + "_gray.jpg", gray);

            // blur image
            Mat blur = matrices.fromSupplier("blur", Mat::new);
            Imgproc.GaussianBlur(gray, blur, new Size(5, 5), 0);
            Imgcodecs.imwrite(file.getName() + "_blur.jpg", blur);

            // Попробовать бинаризацию? Вряд ли, т.к. по цвету зачастую не отличаются, но для подкраски самое то
            Mat binary = matrices.fromSupplier("binary", Mat::new);
            Imgproc.adaptiveThreshold(blur, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);
            Imgcodecs.imwrite(file.getName() + "_binary.jpg", binary);

            // canny edge
            Mat canny = matrices.fromSupplier("canny", Mat::new);
            Imgproc.Canny(blur, canny, 100, 255);
            Imgcodecs.imwrite(file.getName() + "_canny.jpg", canny);

            // find contours
            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = matrices.newMatrix("hierarchy");
            Imgproc.findContours(canny, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

            //draw raw contours
            Mat imgRawContours = matrices.fromSupplier("raw_contours", canny::clone);
            Imgproc.drawContours(imgRawContours, contours, -1, new Scalar(200));
            Imgcodecs.imwrite(file.getName() + "_raw_contours.jpg", imgRawContours);

            // remove out small contours
            List<MatOfPoint> bigContours = contours.stream()
                                                   .filter(contour -> {
                                                       Rect boundingRect = Imgproc.boundingRect(contour);
                                                       return boundingRect.height > heightSomePercent && boundingRect.width > widthSomePercent;
                                                   })
                                                   .collect(Collectors.toList());

            // draw big contours
            Mat imgBigContours = matrices.fromSupplier("big_contours", () -> new Mat(canny.rows(), canny.cols(), canny.type()));
            Imgproc.drawContours(imgBigContours, bigContours, -1, new Scalar(200));
            Imgcodecs.imwrite(file.getName() + "_big_contours.jpg", imgBigContours);

            //draw rectangles
            Mat rects = matrices.fromSupplier("rects", canny::clone);
            bigContours.forEach(contour -> {
                Rect rect = Imgproc.boundingRect(contour);
                Imgproc.rectangle(rects, rect.br(), rect.tl(), new Scalar(500));
            });
            Imgcodecs.imwrite(file.getName() + "_rects.jpg", rects);

            // flood fill big contours
            Mat floodCanny = matrices.fromSupplier("floodCanny", canny::clone);
            //            Imgproc.fillPoly(floodCanny, bigContours, new Scalar(255));
            bigContours.forEach(contour -> Imgproc.fillConvexPoly(floodCanny, contour, new Scalar(255)));

            Mat morphKernel = matrices.fromSupplier("morph_kernel", () -> Imgproc.getStructuringElement(MORPH_OPEN, new Size(3, 3)));
            Imgproc.dilate(floodCanny, floodCanny, morphKernel);
            Imgproc.dilate(floodCanny, floodCanny, morphKernel);
            Imgproc.dilate(floodCanny, floodCanny, morphKernel);
            Imgproc.dilate(floodCanny, floodCanny, morphKernel);
            Imgproc.dilate(floodCanny, floodCanny, morphKernel);
            Imgproc.dilate(floodCanny, floodCanny, morphKernel);
            Imgcodecs.imwrite(file.getName() + "_flood_canny.jpg", floodCanny);

            //find contours again
            List<MatOfPoint> secondContours = new ArrayList<>();
            Mat secondHierarchy = matrices.newMatrix("second_hierarchy");
            Imgproc.findContours(floodCanny, secondContours, secondHierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

            //draw rectangles
            Mat secondRects = matrices.fromSupplier("second_rects", canny::clone);
            secondContours.stream()
                          .map(Imgproc::boundingRect)
                          .filter(rect -> rect.width > cropped.rows() / 100 * 30)
                          .forEach(rect -> {
                              Imgproc.rectangle(secondRects, rect.br(), rect.tl(), new Scalar(500));
                          });
            Imgcodecs.imwrite(file.getName() + "_second_rects.jpg", secondRects);

            //image,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


            /* doesn't work
            //dilate from canny
            Mat dilated = matrices.fromSupplier("dilate", () -> canny.clone());
            for (int i = 0; i < 10; i++) {
                Mat dilateKernel = matrices.fromSupplier("dilate_kernel", () -> Imgproc.getStructuringElement(MORPH_OPEN, new Size(49, 49)));
                Imgproc.dilate(dilated, dilateKernel, dilateKernel);
            }
            Imgcodecs.imwrite(file.getName() + "_dilate.jpg", dilated);*/
        }
    }
}


/*   // erode and dilate
     Mat morph = matrices.fromSupplier("morph", Mat::new);
     Mat morphKernel = matrices.fromSupplier("morph_kernel", () -> Imgproc.getStructuringElement(MORPH_OPEN, new Size(6, 6)));
     Imgproc.morphologyEx(blur, morph, MORPH_OPEN, morphKernel);
     Imgcodecs.imwrite(file.getName() + "_morph.jpg", morph);*/

/*            // flood fill big contours
            Mat floodCanny = matrices.fromSupplier("floodCanny", canny::clone);
//            Imgproc.fillPoly(floodCanny, bigContours, new Scalar(255));
            bigContours.forEach(contour -> Imgproc.fillConvexPoly(floodCanny, contour, new Scalar(255)));
            Imgcodecs.imwrite(file.getName() + "_flood_canny.jpg", floodCanny);*/