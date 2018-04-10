package io.opencv.first.matrixanalysis.quantization;

import io.opencv.first.matrixanalysis.RenameShit;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class QuantizationCheckerShit {

    private static final List<String> IMG_EXTENSIONS = Arrays.asList(".jpg", ".jpeg");
    private static final List<String> IMG_EXTENSIONS_UPPER = IMG_EXTENSIONS.stream()
                                                                           .map(String::toUpperCase)
                                                                           .collect(Collectors.toList());

    public static void main(String[] args) throws IOException {
        List<File> images = Files.walk(Paths.get("D:\\Downloads\\random photos"))
                                 .filter(Files::isRegularFile)
                                 .map(Path::toFile)
                                 .filter(file -> isImage(file.getName()))
                                 .collect(Collectors.toList());

        ShitAggregator shitAggregator = new ShitAggregator();

        images.parallelStream()
              .forEach(shitAggregator::checkFile);
    }

    private static boolean isImage(String fileName) {
        for (String imgExtension : IMG_EXTENSIONS) {
            if (fileName.endsWith(imgExtension)) {
                return true;
            }
        }
        for (String imgExtension : IMG_EXTENSIONS_UPPER) {
            if (fileName.endsWith(imgExtension)) {
                return true;
            }
        }
        return false;
    }
}
