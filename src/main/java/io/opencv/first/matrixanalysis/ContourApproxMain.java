package io.opencv.first.matrixanalysis;

import io.opencv.OpenCvBased;
import io.opencv.util.Matrices;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.opencv.imgcodecs.Imgcodecs.IMREAD_GRAYSCALE;
import static org.opencv.imgproc.Imgproc.RETR_EXTERNAL;
import static org.opencv.imgproc.Imgproc.RETR_TREE;

public class ContourApproxMain extends OpenCvBased {

    public static void main(String[] args) throws IOException {
        File file = new ClassPathResource("images/rectangle.jpg").getFile();
        //        File file = new ClassPathResource("images/IMG_1374.JPG").getFile();

        try (Matrices matrices = new Matrices()) {
            Mat orig_pic = matrices.fromSupplier("orig_pic", () -> Imgcodecs.imread(file.getAbsolutePath(), IMREAD_GRAYSCALE));
            //            Imgproc.Canny(orig_pic, orig_pic, 50, 100);

//            Imgproc.blur(orig_pic, orig_pic, new Size(3, 3));

            List<MatOfPoint> matOfPoints = new ArrayList<>();
            Mat hierarchy = matrices.newMatrix("hierarchy");
            Imgproc.findContours(orig_pic, matOfPoints, hierarchy, RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

            Imgproc.drawContours(orig_pic, matOfPoints, -1, new Scalar(500));
            Imgcodecs.imwrite(file.getName() + "_contours.jpg", orig_pic);

            List<MatOfPoint2f> matOfPoint2fs = new ArrayList<>();
            for (MatOfPoint matOfPoint : matOfPoints) {
                MatOfPoint2f dst = new MatOfPoint2f();
                matOfPoint.convertTo(dst, CvType.CV_32F);
                matOfPoint2fs.add(dst);
            }

            // approx curves
            for (MatOfPoint2f matOfPoint2f : matOfPoint2fs) {
                double epsilon = 0.01 * Imgproc.arcLength(matOfPoint2f, true);
                Imgproc.approxPolyDP(matOfPoint2f, matOfPoint2f, epsilon, true);
            }

            //Convert back to MatOfPoint
            List<MatOfPoint> approxedMatOfPoints = new ArrayList<>();
            for (MatOfPoint2f matOfPoint2f : matOfPoint2fs) {
                approxedMatOfPoints.add(new MatOfPoint(matOfPoint2f.toArray()));
            }

            Mat contours = matrices.fromSupplier("contours", () -> new Mat(orig_pic.size(), 1));
//            Mat contours = matrices.fromSupplier("contours", () -> orig_pic.clone());
//            Imgproc.drawContours(contours, matOfPoints, -1, new Scalar(500));
//            Imgcodecs.imwrite(file.getName() + "_contours.jpg", contours);

            Imgproc.polylines(contours, matOfPoints, true, new Scalar(250));
            Imgcodecs.imwrite(file.getName() + "_polylines.jpg", contours);


            /*vector<Point> ConvexHullPoints =  contoursConvexHull(contours);

    polylines( drawing, ConvexHullPoints, true, Scalar(0,0,255), 2 );
    imshow("Contours", drawing);

    polylines( src, ConvexHullPoints, true, Scalar(0,0,255), 2 );
    imshow("contoursConvexHull", src);
    waitKey();*/
        }
    }
}
