// web/server/faqService.ts
import fs from "fs";
import path from "path";
import { parse } from "csv-parse/sync";
import axios from "axios";
import { Faq } from "@/types";

const faqCache: Record<string, Faq[]> = {};
const DATA_DIR =
  process.env.MFAQ_DATA_DIR ||
  path.resolve(process.cwd(), "..", "data", "mfaq");
const SEMANTIC_ENGINE_URL =
  process.env.SEMANTIC_ENGINE_URL || "http://localhost:8000";

/**
 * Load and cache FAQ entries from the language-specific MFAQ CSV file.
 */
export async function getFaqsByLanguage(language: string): Promise<Faq[]> {
  if (faqCache[language]) return faqCache[language];

  const filePath = path.join(DATA_DIR, `mfaq_${language}_test.csv`);
  if (!fs.existsSync(filePath)) {
    throw new Error(`FAQ dataset not found for language '${language}': ${filePath}`);
  }

  const rawCsv = fs.readFileSync(filePath, "utf-8");
  const records = parse(rawCsv, {
    columns: true,
    skip_empty_lines: true,
    relax_quotes: true,
    relax_column_count: true,
  }) as Array<{ question?: string; answer?: string }>;

  const faqs: Faq[] = records
    .filter((row) => row.question && row.answer)
    .map((row) => ({
      question: String(row.question).trim(),
      answer: String(row.answer).trim(),
    }));

  faqCache[language] = faqs;
  return faqs;
}

/**
 * Query the Python FastAPI Hybrid Bi-Encoder + Cross-Encoder service.
 */
export async function searchFaqs(
  query: string,
  language: string
): Promise<Faq[]> {
  try {
    const resp = await axios.post(`${SEMANTIC_ENGINE_URL}/search`, {
      query,
      lang: language,
      top_k: 5,
    });

    if (Array.isArray(resp.data?.results) && resp.data.results.length > 0) {
      return resp.data.results.map((item: { question: string; answer: string }) => ({
        question: item.question,
        answer: item.answer,
      }));
    }

    const { question, answer } = resp.data;
    return question && answer ? [{ question, answer }] : [];
  } catch (e) {
    console.error("Error searching FAQs via semantic engine:", e);
    return [];
  }
}
