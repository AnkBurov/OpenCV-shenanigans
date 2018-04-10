package io.opencv.first.matrixanalysis;

import io.opencv.OpenCvBased;
import io.opencv.util.Matrices;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static io.opencv.first.matrixanalysis.SamePixelColorMatricsFinder.isMatrixOfSameColor;
import static org.opencv.core.CvType.CV_8UC3;
import static org.opencv.imgcodecs.Imgcodecs.IMREAD_GRAYSCALE;
import static org.opencv.imgproc.Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C;
import static org.opencv.imgproc.Imgproc.ADAPTIVE_THRESH_MEAN_C;
import static org.opencv.imgproc.Imgproc.CHAIN_APPROX_NONE;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2HSV;
import static org.opencv.imgproc.Imgproc.MORPH_OPEN;
import static org.opencv.imgproc.Imgproc.RETR_EXTERNAL;
import static org.opencv.imgproc.Imgproc.RETR_TREE;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY;

@Slf4j
public class RenameShit extends OpenCvBased {

    private static final int MATRIX_SIZE = 20;
    private static final int MAX_DEVIATION = 2;

    //    private static final int IS_WHITE_THRESHOLD = 750;
    private static final int IS_WHITE_THRESHOLD = 240;
    private static final int IS_BLACK_THRESHOLD = 15;

    public static void main(String[] args) throws IOException {
        //        File file = new ClassPathResource("images/14_photoshoped.jpg").getFile();
        //        File file = new ClassPathResource("images/photo17_edited.jpg").getFile();
                        File file = new ClassPathResource("images/618_c56bd967-4aad-4a75-b750-9df320a9d0cd.jpg").getFile();
//                        File file = new ClassPathResource("images/фото ТС 2.JPG").getFile();
//        File file = new ClassPathResource("images/132_029eac4e-05b7-477d-b903-996bc61e622f.jpg").getFile();
//                        File file = new ClassPathResource("images/IMG_5023_1600.png").getFile();
        //                File file = new ClassPathResource("images/IMG_1374.JPG").getFile();

        handleFile(file, MATRIX_SIZE, MAX_DEVIATION, IS_WHITE_THRESHOLD, IS_BLACK_THRESHOLD, 0.9);
    }

    public static void handleFile(File file, int matrixSize, int maxDeviation, int whiteThreshold, int blackThreshold, double percentToSave) {
        log.info("Starting analyzing of " + file.getAbsolutePath());

        //todo try sharpen image
        try (Matrices matrices = new Matrices()) {
            Mat orig_notCropped = matrices.fromSupplier("orig", () -> Imgcodecs.imread(file.getAbsolutePath()));
            Mat orig = cropUpperBound(matrices, orig_notCropped, percentToSave);
            Mat orig_pic = matrices.fromSupplier("orig_pic", orig::clone);

            Mat gauss = matrices.newMatrix("gauss");
            Imgproc.GaussianBlur(orig, gauss, new Size(5, 5), 0.1);
            writeImage(gauss, file, "_gauss.jpg");

            Mat canny = matrices.newMatrix("canny");
            Imgproc.Canny(gauss, canny, 3, 10);
//                        Imgproc.blur(canny, canny,  new Size(2, 2));
            //            Imgproc.GaussianBlur(canny, canny,, 5);
            //            Imgproc.GaussianBlur(canny, canny, new Size(3, 3), 1);
            writeImage(canny, file, "_canny.jpg");

            Mat erode = matrices.newMatrix("erode");
            Mat dilateKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
                                                             new Size(4, 4));
            Imgproc.dilate(canny, erode, dilateKernel);
            writeImage(erode, file, "_erode.jpg");

            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = matrices.newMatrix("hierarchy");
            Imgproc.findContours(erode, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
            System.out.println("Number of contours is: " + contours.size());

            Map<Rect, MatOfPoint> rectMatOfPointMap = new ConcurrentHashMap<>();
            // means contour has no children
            IntStream.range(0, contours.size())
                     .parallel()
                     .forEach(i -> {
                         MatOfPoint contour = contours.get(i);
                         Rect rect = Imgproc.boundingRect(contour);
                         if (rect.width > matrixSize && rect.height > matrixSize) {
                             if (hierarchy.get(0, i)[2] == -1) { // means contour has no children
                                 rectMatOfPointMap.put(rect, contour);

                                 Imgproc.rectangle(orig, rect.br(), rect.tl(), new Scalar(500));
                             }
                         }
                     });
            writeImage(orig, file, "_orig.jpg");

            System.out.println(rectMatOfPointMap.size());

            Mat contoursMat = matrices.fromSupplier("contours", () -> new Mat(new Size(orig.width(), orig.height()), CV_8UC3));
            Imgproc.drawContours(contoursMat, new ArrayList<>(rectMatOfPointMap.values()), -1, new Scalar(500));
            writeImage(contoursMat, file, "_contours.jpg");

            Map<Rect, MatOfPoint> contoursWithSameMatrices = detectSameMatrices(orig, rectMatOfPointMap, matrixSize);
            //            Imgcodecs.imwrite(file.getName() + "_analyzed.jpg", orig);

            List<Point> pointsOfSameColorContours = new ArrayList<>();

            for (Map.Entry<Rect, MatOfPoint> entry : contoursWithSameMatrices.entrySet()) {
                Rect rect = entry.getKey();
                MatOfPoint contour = entry.getValue();
                MatOfPoint2f curve = new MatOfPoint2f(contour.toArray());

                List<Point> pointsOfContourArea = Collections.synchronizedList(new ArrayList<>());
                IntStream.rangeClosed(rect.y, rect.y + rect.height)
                         .parallel()
                         .forEach(row -> IntStream
                                 .rangeClosed(rect.x, rect.x + rect.width)
                                 .parallel()
                                 .forEach(column -> {
                                     Point point = new Point(column, row);
                                     boolean belongsToContour = Imgproc.pointPolygonTest(curve, point, false) >= 0;
                                     if (belongsToContour) {
                                         pointsOfContourArea.add(point);
                                     }
                                 }));

                System.out.println("Points inside of contour: " + pointsOfContourArea.size());

                List<Double> aggregatedPixelValues = new ArrayList<>();
                for (Point point : pointsOfContourArea) {
                    double[] pixel = orig_pic.get((int) point.y, (int) point.x);
                    Arrays.stream(pixel).forEach(aggregatedPixelValues::add);
                }

                //calculate standard variance
                Statistics statistics = new Statistics(aggregatedPixelValues.toArray(new Double[aggregatedPixelValues.size()]));
                double stdDev = statistics.getStdDev();
                if (stdDev < maxDeviation) {

                    double median = statistics.median();

                    // check if contour is white inside
                    System.out.println("!!!!!!! median is " + median);

                    if (median > whiteThreshold) {
                        ExtremeShit extremeShit = getExtremeShit(pointsOfContourArea); // null check
                        Double medianOfAdjacentPixels = getMedianOfAdjacentPixels(orig_pic, extremeShit, 2);

                        if (medianOfAdjacentPixels < whiteThreshold) { //todo remove?
                            pointsOfSameColorContours.addAll(pointsOfContourArea);
                        }
                    } else if (median < blackThreshold) {
                        ExtremeShit extremeShit = getExtremeShit(pointsOfContourArea); // null check
                        Double medianOfAdjacentPixels = getMedianOfAdjacentPixels(orig_pic, extremeShit, 2);

                        if (medianOfAdjacentPixels > blackThreshold) { //todo remove?
                            pointsOfSameColorContours.addAll(pointsOfContourArea);
                        }
                    } else {
                        // check if color difference with adjacent pixels is not too small
                        //                        ExtremeShit extremeShit = getExtremeShit(pointsOfContourArea); // null check
                        //                        Double medianOfAdjacentPixels = getMedianOfAdjacentPixels(orig_pic, extremeShit, 5); //tune maybe recursive

                        //                        System.out.println("Median " + statistics.median() + " medianOfAdjacentPixels " + medianOfAdjacentPixels);
                        //                        if (statistics.median() - medianOfAdjacentPixels >= 1
                        //                                || statistics.median() - medianOfAdjacentPixels <= -1) {
                        pointsOfSameColorContours.addAll(pointsOfContourArea);
                        //                        }
                    }
                }
            }

            for (Point sameColorPoint : pointsOfSameColorContours) {
                orig_pic.put((int) sameColorPoint.y, (int) sameColorPoint.x, 0, 255, 0);
            }
            if (!pointsOfSameColorContours.isEmpty()) {
                Imgcodecs.imwrite(file.getName() + "_analyzed_similar.jpg", orig_pic);
            }
        }
        log.info("Ended analyzing of " + file.getAbsolutePath());
    }

    private static Map<Rect, MatOfPoint> detectSameMatrices(Mat orig, Map<Rect, MatOfPoint> rectMatOfPointMap, int matrixSize) {
        Map<Rect, MatOfPoint> contoursWithSameMatrices = new ConcurrentHashMap<>();

        List<Map.Entry<Integer, Integer>> rowColumns = Collections.synchronizedList(new ArrayList<>());

        // local cache of already analyzed matrices
        Map<Point, Boolean> analyzedMatricesByStartPoint = new ConcurrentHashMap<>();

        // for each rectangle shit
        for (Map.Entry<Rect, MatOfPoint> entry : rectMatOfPointMap.entrySet()) {
            Rect rect = entry.getKey();
            MatOfPoint contour = entry.getValue();

            // analyze each rectangle shit of interest for same matrices
            IntStream.rangeClosed(rect.y, rect.y + rect.height)
                     .parallel()
                     .forEach(row -> IntStream
                             .rangeClosed(rect.x, rect.x + rect.width)
                             .parallel()
                             .forEach(column -> {
                                 analyzedMatricesByStartPoint.computeIfAbsent(new Point(column, row), point -> {
                                     int matrixRow = (int) point.y;
                                     int matrixColumn = (int) point.x;
                                     Boolean matrixOfSameColor = isMatrixOfSameColor(orig, matrixRow, matrixColumn, matrixSize);
                                     if (matrixOfSameColor) {
                                         //                            System.out.println("Matrics starts on row | column " + matrixRow + "|" + matrixColumn + " contains only same color");
                                         rowColumns.add(new AbstractMap.SimpleEntry<>(matrixRow, matrixColumn));

                                         contoursWithSameMatrices.put(rect, contour);
                                     }
                                     return matrixOfSameColor;
                                 });
                             }));
        }

        // rectangle the similar matrices and update the image
        rowColumns.parallelStream().forEach(rowColumn -> {
            Point startingPoing = new Point(rowColumn.getValue(), rowColumn.getKey());
            Point endingPoint = new Point(rowColumn.getValue() + matrixSize, rowColumn.getKey() + matrixSize);
            Imgproc.rectangle(orig, startingPoing, endingPoint, new Scalar(0, 255, 0));
        });

        return contoursWithSameMatrices;
    }

    /*//thresholding
            Mat threshold = matrices.newMatrix("threshold");
            Imgproc.adaptiveThreshold(orig, threshold, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
            writeImage(threshold, file, "_threshold.jpg");*/

    /*
            //https://stackoverflow.com/questions/18627970/adaptive-threshold-with-blurry-image
            Mat hsv = matrices.newMatrix("hsv");
            Imgproc.cvtColor(orig, hsv, COLOR_BGR2HSV);
            writeImage(hsv, file, "_hsv.jpg");*/

    public static void writeImage(Mat orig, File file, String name) {
        if (false) {
            Imgcodecs.imwrite(file.getName() + name, orig);
        }
    }

    private static ExtremeShit getExtremeShit(List<Point> points) {
        if (points.isEmpty()) {
            return null;
        }
        Point leftMost;
        Point topMost;
        Point rightMost;
        Point bottomMost;

        List<Point> sortedByX = points.stream()
                                      .sorted(Comparator.comparing(it -> it.x))
                                      .collect(Collectors.toList());
        leftMost = sortedByX.get(0);
        rightMost = sortedByX.get(sortedByX.size() - 1);

        List<Point> sortedByY = points.stream()
                                      .sorted(Comparator.comparing(it -> it.y))
                                      .collect(Collectors.toList());
        bottomMost = sortedByY.get(0);
        topMost = sortedByY.get(sortedByY.size() - 1);
        return ExtremeShit.builder()
                          .leftMost(leftMost)
                          .topMost(topMost)
                          .rightMost(rightMost)
                          .bottomMost(bottomMost)
                          .build();
    }

    private static Double getMedianOfAdjacentPixels(Mat mat, ExtremeShit extremeShit, int numberOfSteps) {
        double[] leftMostNeighbourPixel = mat.get((int) extremeShit.getLeftMost().y, (int) extremeShit.getLeftMost().x - numberOfSteps);
        double[] topMostNeighbourPixel = mat.get((int) extremeShit.getTopMost().y + numberOfSteps, (int) extremeShit.getTopMost().x);
        double[] rightMostNeighbourPixel = mat.get((int) extremeShit.getRightMost().y, (int) extremeShit.getRightMost().x + numberOfSteps);
        double[] bottomMostNeighbourPixel = mat.get((int) extremeShit.getBottomMost().y - numberOfSteps, (int) extremeShit.getBottomMost().x);

        Double[] aggregatedPixelValues = Stream.of(leftMostNeighbourPixel, topMostNeighbourPixel, rightMostNeighbourPixel, bottomMostNeighbourPixel)
                                               .filter(it -> it != null)
                                               .flatMapToDouble(Arrays::stream)
                                               .boxed()
                                               .toArray(Double[]::new);
        return new Statistics(aggregatedPixelValues).median();
    }

    private static Mat cropUpperBound(Matrices matrices, Mat source, double percentToSave) {
//        Double width = source.cols() * percentToSave;
//        Double height = source.rows() * percentToSave;
//        int x = source.cols() - width.intValue();
//        int y = source.rows() - height.intValue();
//
//        Rect rectCrop = new Rect(x, y, width.intValue() - x, height.intValue() - y);
//        return matrices.fromSupplier("croped", () -> new Mat(source, rectCrop));
        Double width = (double) source.cols();
        Double height = source.rows() * percentToSave;
        int x = 0;
        int y = source.rows() - height.intValue();

        Rect rectCrop = new Rect(x, y, width.intValue(), height.intValue());
        return matrices.fromSupplier("croped", () -> new Mat(source, rectCrop));
    }

}
