package edu.berkeley.nlp.lm.io;

import edu.berkeley.nlp.lm.ConfigOptions;
import edu.berkeley.nlp.lm.StringWordIndexer;
import edu.berkeley.nlp.lm.collections.Iterators;
import edu.berkeley.nlp.lm.util.Pair;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class KneserNeyFromTextReaderTest
{

	@Test
	public void testBigram() {
		doTest("tiny_test_bigram", new double[] { 0.75f, 0.33333f });
	}

	@Test
	public void testTrigram() {
		doTest("tiny_test_trigram", new double[] { 0.75f, 0.6f, 0.6f });
	}

	@Test
	public void testFivegram() {
		doTest("tiny_test_fivegram", new double[] { 0.4f, 0.5f, 0.5f, 0.538462f, 0.454545f });
	}

	@Test
	public void testBig() {
		doTest("big_test", new double[] { 0.755639f, 0.891934f, 0.944268f, 0.955941f, 0.359436f });
	}

	private void doTest(final String prefix, final double[] discounts) {
		final StringWordIndexer wordIndexer = new StringWordIndexer();
		final int order = discounts.length;
		wordIndexer.setStartSymbol("<s>");
		wordIndexer.setEndSymbol("</s>");
		wordIndexer.setUnkSymbol("<unk>");
		final String txtFile = FileUtils.getFile(prefix + ".txt").getPath();
		final File goldArpaFile = FileUtils.getFile(prefix + ".arpa");
		final StringWriter stringWriter = new StringWriter();
		final TextReader<String> reader = new TextReader<>(List.of(txtFile), wordIndexer);
		final ConfigOptions opts = new ConfigOptions();
		opts.kneserNeyDiscounts = discounts;
		opts.kneserNeyMinCounts = new double[] { 0, 0, 0, 0, 0, 0, 0 };
		final KneserNeyLmReaderCallback<String> kneserNeyReader = new KneserNeyLmReaderCallback<>(wordIndexer, order, opts);
		reader.parse(kneserNeyReader);
		KneserNeyFileWritingLmReaderCallback<String> kneserNeyFileWriter = new KneserNeyFileWritingLmReaderCallback<>(new PrintWriter(stringWriter),
			wordIndexer);
		kneserNeyReader.parse(kneserNeyFileWriter);

		final List<String> arpaLines = new ArrayList<>(List.of(stringWriter.toString().split("\n")));
		sortAndRemoveBlankLines(arpaLines);
		final List<String> goldArpaLines = getLines(goldArpaFile);
		sortAndRemoveBlankLines(goldArpaLines);
		compareLines(arpaLines, goldArpaLines);
	}

	private void compareLines(final List<String> arpaLines, final List<String> goldArpaLines) {
		assertEquals(arpaLines.size(), goldArpaLines.size());
		for (final Pair<String, String> lines : Iterators.able(Iterators.zip(arpaLines.iterator(), goldArpaLines.iterator()))) {
			final String testLine = lines.getFirst().trim();
			final String goldLine = lines.getSecond().trim();
			if (goldLine.startsWith("-")) {
				assertTrue(testLine.startsWith("-"));
				final String[] testSplit = testLine.split("\t");
				final String[] goldSplit = goldLine.split("\t");
				assertEquals(testSplit.length, goldSplit.length);
				assertTrue(testSplit.length == 2 || testSplit.length == 3);
				assertEquals(testSplit[1], goldSplit[1]);
				assertEquals(Double.parseDouble(testSplit[0]), Double.parseDouble(goldSplit[0]), 1e-3);
				if (testSplit.length == 3) {
					assertEquals(Double.parseDouble(testSplit[2]), Double.parseDouble(goldSplit[2]), 1e-3);
				}

			} else {
				assertEquals(testLine, goldLine);
			}
		}
	}

	private List<String> getLines(final File goldArpaFile) {
		final List<String> ret = new ArrayList<>();
		try {
			for (final String line : Iterators.able(IOUtils.lineIterator(goldArpaFile.getAbsolutePath()))) {
				ret.add(line);
			}
			return ret;
		} catch (final IOException e) {
			throw new RuntimeException(e);

		}
	}

	private void sortAndRemoveBlankLines(final List<String> arpaLines) {
		arpaLines.sort((arg0, arg1) -> {
			final String[] split1 = arg0.split("\t");
			final String[] split2 = arg1.split("\t");
			final int x = Double.compare(split1.length, split2.length);
			if (x != 0) return x;
			if (split1.length > 1) return split1[1].compareTo(split2[1]);
			return split1[0].compareTo(split2[0]);
		});
		for (int i = arpaLines.size() - 1; i >= 0; i--) {
			if (arpaLines.get(i).trim().isEmpty()) arpaLines.remove(i);
		}
	}

}
