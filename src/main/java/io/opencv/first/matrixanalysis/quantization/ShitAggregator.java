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
            List<Integer> connectedQuantizationValues = quantizationShit.parseQuantizationTables(file)
                                                                        .stream()
                                                                        .flatMap(Collection::stream)
                                                                        .collect(Collectors.toList());

            String joinedValues = StringUtils.join(connectedQuantizationValues, "");

            MessageDigest digest = MessageDigest.getInstance("MD5");
            digest.update(joinedValues.getBytes("UTF-8"));
            String hash = Hex.encodeHexString(digest.digest());

            jdbcTemplate.update("insert into DEVICE_HASH (MODEL, QUANTIZATION_HASH) values (?, ?)", model.orElse("UNKNOWN"), hash);

        } catch (IOException | NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        }
    }

    public static void main(String[] args) throws NoSuchAlgorithmException, IOException {
        File file = new ClassPathResource("images/paint.jpg").getFile();

        Optional<String> model = metadataParser.parseModel(file.toPath());

        List<List<Integer>> quantizationTables = quantizationShit.parseQuantizationTables(file);

        List<Integer> connectedQuantizationValues = quantizationTables
                .stream()
                .flatMap(Collection::stream)
                .collect(Collectors.toList());

        String joinedValues = StringUtils.join(connectedQuantizationValues, "");

        MessageDigest digest = MessageDigest.getInstance("MD5");
        digest.update(joinedValues.getBytes("UTF-8"));
        String hash = Hex.encodeHexString(digest.digest());
        System.out.println(hash);
    }
}
