package edu.berkeley.nlp.lm.io;

import java.io.File;
import java.util.Arrays;

import edu.berkeley.nlp.lm.ArrayEncodedNgramLanguageModel;
import edu.berkeley.nlp.lm.ArrayEncodedProbBackoffLm;
import edu.berkeley.nlp.lm.ConfigOptions;
import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel;
import edu.berkeley.nlp.lm.ContextEncodedProbBackoffLm;
import edu.berkeley.nlp.lm.StringWordIndexer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MissingEntryTest
{

	private static final double TOL = 1e-5;

	public static final String BIG_TEST_ARPA = "missing_test_fourgram.arpa";

	@Test
	public void testArrayEncoded() {

		final ArrayEncodedProbBackoffLm<String> lm = getLm(false);
		testArrayEncodedLogProb(lm);
	}

	@Test
	public void testCompressedEncoded() {

		final ArrayEncodedProbBackoffLm<String> lm = getLm(true);
		testArrayEncodedLogProb(lm);
	}

	@Test
	public void testContextEncoded() {

		final ContextEncodedProbBackoffLm<String> lm = getContextEncodedLm();
		testContextEncodedLogProb(lm);
	}

	private ContextEncodedProbBackoffLm<String> getContextEncodedLm() {
		final File lmFile = FileUtils.getFile(BIG_TEST_ARPA);
		final ConfigOptions configOptions = new ConfigOptions();
		configOptions.unknownWordLogProb = 0.0f;
		return LmReaders.readContextEncodedLmFromArpa(lmFile.getPath(), new StringWordIndexer(), configOptions,
													  Integer.MAX_VALUE);
	}

	private ArrayEncodedProbBackoffLm<String> getLm(boolean compress) {
		final File lmFile = FileUtils.getFile(BIG_TEST_ARPA);
		final ConfigOptions configOptions = new ConfigOptions();
		configOptions.unknownWordLogProb = 0.0f;
		return LmReaders.readArrayEncodedLmFromArpa(lmFile.getPath(), compress, new StringWordIndexer(), configOptions,
													Integer.MAX_VALUE);
	}

	public static void testArrayEncodedLogProb(final ArrayEncodedNgramLanguageModel<String> lm_) {

		assertEquals(lm_.getLogProb(Arrays.asList("This another test is".split(" "))), -0.67443009, TOL);
		assertEquals(lm_.getLogProb(Arrays.asList("another test sentence.".split(" "))), -0.07443009, TOL);
		assertEquals(lm_.getLogProb(Arrays.asList("is another test".split(" "))), -0.1366771, TOL);
		assertEquals(lm_.getLogProb(Arrays.asList("another test".split(" "))), -0.60206 + -0.2218488, TOL);
	}

	public static void testContextEncodedLogProb(final ContextEncodedNgramLanguageModel<String> lm_) {

		assertEquals(lm_.getLogProb(Arrays.asList("This another test is".split(" "))), -0.67443009, TOL);
		assertEquals(lm_.getLogProb(Arrays.asList("another test sentence.".split(" "))), -0.07443009, TOL);
		assertEquals(lm_.getLogProb(Arrays.asList("is another test".split(" "))), -0.1366771, TOL);
		assertEquals(lm_.getLogProb(Arrays.asList("another test".split(" "))), -0.60206 + -0.2218488, TOL);
	}
}
