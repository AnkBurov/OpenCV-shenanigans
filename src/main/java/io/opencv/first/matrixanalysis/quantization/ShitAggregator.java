package io.opencv.first.matrixanalysis.quantization;

import org.apache.commons.codec.binary.Hex;
import org.apache.commons.lang.StringUtils;
import org.springframework.core.io.ClassPathResource;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.datasource.SimpleDriverDataSource;

import javax.sql.DataSource;
import java.io.File;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Collection;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

public class ShitAggregator {

    private static MetadataParser metadataParser = new MetadataParser();

    private static QuantizationShit quantizationShit = new QuantizationShit();

    private static DataSource dataSource = new SimpleDriverDataSource(org.h2.Driver.load(), "jdbc:h2:", "sa", "");

    private static JdbcTemplate jdbcTemplate = new JdbcTemplate(dataSource);

    public void checkFile(File file) {
        try {
            Optional<String> model = metadataParser.parseModel(file.toPath());

            List<List<Integer>> tables = quantizationShit.parseQuantizationTablesUsingImageIO(file);

            List<Integer> connectedQuantizationValues = tables
                    .stream()
                    .flatMap(Collection::stream)
                    .collect(Collectors.toList());

            String joinedValues = StringUtils.join(connectedQuantizationValues, "");

            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            digest.update(joinedValues.getBytes("UTF-8"));
            String hash = Hex.encodeHexString(digest.digest());

            jdbcTemplate.update("insert into DEVICE_HASH (MODEL, QUANTIZATION_HASH, FILE) values (?, ?, ?)", model.orElse("UNKNOWN"), hash, file.getName());

        } catch (IOException | NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        }
    }

    public static void main(String[] args) throws NoSuchAlgorithmException, IOException {
        File file = new ClassPathResource("images/shit/paint.jpg").getFile();

        Optional<String> model = metadataParser.parseModel(file.toPath());

        List<List<Integer>> quantizationTables = quantizationShit.parseQuantizationTablesUsingImageIO(file);

        List<Integer> connectedQuantizationValues = quantizationTables
                .stream()
                .flatMap(Collection::stream)
                .collect(Collectors.toList());

        String joinedValues = StringUtils.join(connectedQuantizationValues, "");

        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        digest.update(joinedValues.getBytes("UTF-8"));
        String hash = Hex.encodeHexString(digest.digest());
        System.out.println(hash);
        // iphone 7 c2a08ac61467ca3fc21052744789f406991161ca7f78f4e622ef3e9e6810b02c
        // iphone 7 80d1db7905b722a486fe7d9a57d52ba6f14e221261474d80350cc73a65d39a45
        // paint ee420bc5a55c37e2d86477c4db4f4c8ac9869d17574e8c24106419207acd863d
        // paint ee420bc5a55c37e2d86477c4db4f4c8ac9869d17574e8c24106419207acd863d
    }
}
