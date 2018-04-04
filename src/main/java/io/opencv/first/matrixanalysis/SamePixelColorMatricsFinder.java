package io.opencv.first.matrixanalysis;

import io.opencv.OpenCvBased;
import io.opencv.util.Matrices;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.opencv.imgproc.Imgproc.MORPH_OPEN;
import static org.opencv.imgproc.Imgproc.RETR_EXTERNAL;

@Slf4j
public class SamePixelColorMatricsFinder extends OpenCvBased {

    private static final int MATRIX_SIZE = 20;

    public static void main(String[] args) throws IOException {
        //        File file = new ClassPathResource("images/IMG_1151_photoshoped.jpg").getFile();
//        File file = new ClassPathResource("images/white_black.png").getFile();
//        File file = new ClassPathResource("images/14_photoshoped.jpg").getFile();
//        File file = new ClassPathResource("images/14.jpg").getFile();
//        File file = new ClassPathResource("images/d7fc888s-960.jpg").getFile();
//        File file = new ClassPathResource("images/IMG_1151.jpg").getFile();
        File file = new ClassPathResource("images/spacex_photoshoped.jpg").getFile();
//        File file = new ClassPathResource("images/carsgraz_001.bmp").getFile();
//        File file = new ClassPathResource("images/spacex.jpg").getFile();

        handleFile(file, MATRIX_SIZE, 0.85);
    }

    public static void handleFile(File file, int matrixSize, double percentToSaveWhenCroping) {
        log.info("Starting analyzing of " + file.getAbsolutePath());
        try (Matrices matrices = new Matrices()) {
            Mat orig_pic = matrices.fromSupplier("orig_pic", () -> Imgcodecs.imread(file.getAbsolutePath()));
            Mat orig = cropImage(matrices, orig_pic, percentToSaveWhenCroping);
            Mat dst = matrices.fromSupplier("new", orig::clone);

//            Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGR2GRAY);

            Imgproc.GaussianBlur(dst, dst, new Size(3, 3), 3);

            Mat kernel = matrices.fromSupplier("whatever", () -> Imgproc.getStructuringElement(MORPH_OPEN, new Size(6, 6)));
//            Imgproc.dilate(dst, dst, kernel);
            Imgproc.morphologyEx(dst, dst, MORPH_OPEN, kernel);

            Mat canny = matrices.fromSupplier("canny", dst::clone);
            Imgproc.Canny(dst, canny, 50, 100);
            //             uncomment if needed
//            Imgcodecs.imwrite(file.getName() + "_canny.jpg", canny);

            List<MatOfPoint> matOfPoints = new ArrayList<>();
            Mat hierarchy = matrices.newMatrix("hierarchy");
            Imgproc.findContours(canny, matOfPoints, hierarchy, RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);


            //tesseract OCR
            //{60, 42, 75x39}
            List<Rect> rectsOfInterest = new ArrayList<>();

            Mat mat = new Mat(canny.rows(), canny.cols(), 1);
            for (MatOfPoint matOfPoint : matOfPoints) {
                Rect rect = Imgproc.boundingRect(matOfPoint);
                if (rect.width > matrixSize && rect.height > matrixSize) {
                    Imgproc.rectangle(mat, rect.br(), rect.tl(), new Scalar(500));
                    rectsOfInterest.add(rect);
                }
            }

//            Imgproc.drawContours(mat, matOfPoints, -1, new Scalar(500));

//             uncomment if needed
//            Imgproc.drawContours(mat, matOfPoints,  -1, new Scalar(500), 1, 1, hierarchy, 1 , new Point(0,0));
//            Imgcodecs.imwrite(file.getName() + "_contours.jpg", mat);


//            Imgproc.cvtColor(dst, dst, Imgproc.COLOR_GRAY2BGR);

            List<Map.Entry<Integer, Integer>> rowColumns = new ArrayList<>();

            // local cache of already analyzed matrices
            Map<Point, Boolean> analyzedMatricesByStartPoint = new HashMap<>();

            // for each rectangle shit
            for (Rect rect : rectsOfInterest) {

                // analyze each rectangle shit of interest for same matrices
                for (int row = rect.y; row <= rect.y + rect.height; row++) {
                    for (int column = rect.x; column <= rect.x + rect.width; column++) {
                        analyzedMatricesByStartPoint.computeIfAbsent(new Point(column, row), point -> {
                            int matrixRow = (int) point.y;
                            int matrixColumn = (int) point.x;
                            Boolean matrixOfSameColor = isMatrixOfSameColor(orig, matrixRow, matrixColumn, matrixSize);
                            if (matrixOfSameColor) {
                                System.out.println("Matrics starts on row | column " + matrixRow + "|" + matrixColumn + " contains only same color");
                                rowColumns.add(new AbstractMap.SimpleEntry<>(matrixRow, matrixColumn));
                            }
                            return matrixOfSameColor;
                        });
                    }
                }
            }

            // rectangle the similar matrices and update the image
            for (Map.Entry<Integer, Integer> rowColumn : rowColumns) {
                Point startingPoing = new Point(rowColumn.getValue() - 1, rowColumn.getKey() - 1);
                Point endingPoint = new Point(rowColumn.getValue() - 1 + matrixSize, rowColumn.getKey() - 1 + matrixSize);
                Imgproc.rectangle(orig, startingPoing, endingPoint, new Scalar(500));
            }

            if (!rowColumns.isEmpty()) {
                Imgcodecs.imwrite(file.getName() + "_analyzed.jpg", orig);
            }
        }
        log.info("Ended analyzing of " + file.getAbsolutePath());
    }

    private static Mat cropImage(Matrices matrices, Mat source, double percentToSave) {
        Double width = source.cols() * percentToSave;
        Double height = source.rows() * percentToSave;
        int x = source.cols() - width.intValue();
        int y = source.rows() - height.intValue();

        Rect rectCrop = new Rect(x, y, width.intValue() - x, height.intValue() - y);
        return matrices.fromSupplier("croped", () -> new Mat(source, rectCrop));
    }

    public static Boolean isMatrixOfSameColor(Mat image, int row, int column, int matrixSize) {
        List<double[]> rowStartingPixels = new ArrayList<>();
        for (int matrixRow = 0; matrixRow < matrixSize; matrixRow++) {
            double[] rowStartingPixel = image.get(row + matrixRow, column);
            Boolean isSameColor = areNextColumnsInMatrixOfSameColor(image, rowStartingPixel,
                    false, row + matrixRow, column, matrixSize - 1);
            if (!Boolean.TRUE.equals(isSameColor)) {
                return false;
            }
            rowStartingPixels.add(rowStartingPixel);
        }
        //  filter values equal to other values in collection and check that all values are equal
        return rowStartingPixels.stream()
                .filter(pixel -> rowStartingPixels.stream()
                        .filter(comparingPixel -> Arrays.equals(pixel, comparingPixel))
                        .count() == rowStartingPixels.size()
                )
                .count() == rowStartingPixels.size();
    }

    private static Boolean areNextColumnsInMatrixOfSameColor(Mat image, double[] previousPixel,
                                                             boolean isSimilar, int row, int column, int depth) {
        if (depth == 0) {
            return isSimilar;
        }

        double[] nextPixel = image.get(row, column);
        if (nextPixel == null) {
            return null;
        }
        if (Arrays.equals(previousPixel, nextPixel)) {
            return areNextColumnsInMatrixOfSameColor(image, nextPixel, true, row, column + 1, depth - 1);
        }
        return false;
    }
}
