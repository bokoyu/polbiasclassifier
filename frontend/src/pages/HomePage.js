import React, { useState } from "react";
import {
  Box,
  Heading,
  Textarea,
  Button,
  Text,
  useToast,
} from "@chakra-ui/react";

import PageCard from "../components/PageCard";
import MetricBlock from "../components/MetricBlock";
import ErrorAlert from "../components/ErrorAlert";


function parseBiasSection(str) {
  const biasRegex = /\[BIAS:\s+(Neutral|Biased)\s+\(([\d.]+)\)\]/;
  const match = str.match(biasRegex);
  return match ? { label: match[1], confidence: match[2] } : null;
}

function parseLeaningSection(str) {
  const leanRegex = /\[LEANING:\s+(Left|Right|Center)\s+\(([\d.]+)\)\]/;
  const match = str.match(leanRegex);
  return match ? { label: match[1], confidence: match[2] } : null;
}

function parsePrediction(rawStr) {
  const bias = parseBiasSection(rawStr);
  const leaning = parseLeaningSection(rawStr);
  return {
    biasLabel: bias?.label || null,
    biasConf: bias?.confidence || null,
    leaningLabel: leaning?.label || null,
    leaningConf: leaning?.confidence || null,
  };
}

export default function HomePage() {
  const toast = useToast();

  const [inputText, setInputText] = useState("");
  const [predictionRaw, setPredictionRaw] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setError(null);
    setPredictionRaw(null);
    setLoading(true);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: inputText }),
      });

      if (!response.ok) {
        const msg = await response.json();
        throw new Error(msg.error || "Request failed");
      }

      const data = await response.json();
      setPredictionRaw(data.prediction);
    } catch (err) {
      setError(err.message);
      toast({
        title: "Error",
        description: err.message,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setLoading(false);
    }
  };

  const { biasLabel, biasConf, leaningLabel, leaningConf } = parsePrediction(
    predictionRaw || ""
  );

  const biasMetrics =
    biasLabel && biasConf ? { [biasLabel]: Number(biasConf) } : {};
  const leaningMetrics =
    leaningLabel && leaningConf ? { [leaningLabel]: Number(leaningConf) } : {};

  return (
    <PageCard>
      <Heading size="md" mb={6} textAlign="center">
        Political Bias Classifier
      </Heading>

      <Textarea
        placeholder="Paste an article or tweet…"
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
        minH="7rem"
        mb={4}
        resize="vertical"
      />

      <Button
        colorScheme="blue"
        width="100%"
        onClick={handlePredict}
        isLoading={loading}
        isDisabled={!inputText.trim()}
      >
        Predict
      </Button>

      {error && <ErrorAlert message={error} />}

      {predictionRaw && (
        <Box mt={6} display="flex" flexDir={{ base: "column", sm: "row" }} gap={6}>
          <MetricBlock title="Bias" metrics={biasMetrics} />
          {biasLabel === "Biased" && (
            <MetricBlock title="Leaning" metrics={leaningMetrics} />
          )}
        </Box>
      )}
    </PageCard>
  );
}