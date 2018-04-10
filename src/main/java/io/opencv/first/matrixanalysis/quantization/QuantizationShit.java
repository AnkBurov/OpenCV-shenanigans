package io.opencv.first.matrixanalysis.quantization;

import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.StringReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.StringJoiner;

public class QuantizationShit {
    public static void main(String[] args) throws IOException {
        File file = new ClassPathResource("images/not_paint.jpg").getFile();

        try (FileInputStream stream = new FileInputStream(file)) {
            if (!isJpeg(stream)) {
                throw new IllegalArgumentException("It's not JPEG");
            }

            List<List<Integer>> quantizationTables = getQuantizationTables(stream);
            for (List<Integer> quantizationTable : quantizationTables) {
                String s = quantizationTable.toString();
                Files.write(Paths.get("table3.txt"), s.getBytes(), StandardOpenOption.APPEND);
                Files.write(Paths.get("table3.txt"), "\n".getBytes(), StandardOpenOption.APPEND);
            }
        }
    }

    public List<List<Integer>> parseQuantizationTables(File file) throws IOException {
        try (FileInputStream stream = new FileInputStream(file)) {
            if (!isJpeg(stream)) {
                throw new IllegalArgumentException(file + " It's not JPEG");
            }

            return getQuantizationTables(stream);
        }
    }

    private static boolean isJpeg(InputStream stream) throws IOException {
        String buffer = "";
        buffer += Integer.toHexString(stream.read());
        buffer += Integer.toHexString(stream.read());

        return buffer.equalsIgnoreCase("FFD8");
    }

    private static List<List<Integer>> getQuantizationTables(InputStream stream) throws IOException {
        List<List<Integer>> quantizationTables = new ArrayList<>();

        int firstByte = -100;
        int secondByte = -100;
        do {
            firstByte = stream.read();

            String firstByteHex = Integer.toHexString(firstByte);
            if (firstByteHex.equalsIgnoreCase("FF")) {
                secondByte = stream.read();
                String secondByteHex = Integer.toHexString(secondByte);

                String hexValue = firstByteHex + secondByteHex;

                if (hexValue.equalsIgnoreCase("FFDB")) {
                    int lengthOfBlock = stream.read() + stream.read();
                    int qtInformation = stream.read();

                    // read quantization table
                    List<Integer> table = new ArrayList<>();
                    // lengthOfBlock is 2 bytes, qtInformation is 1 byte
                    for (int i = 0; i < lengthOfBlock - 3; i++) {
                        table.add(stream.read());
                    }
                    quantizationTables.add(table);
                } else if (hexValue.equalsIgnoreCase("FFC0")) {
                    break;
                }
            }
        } while (firstByte != -1 && secondByte != -1);

        return quantizationTables;
    }
}
