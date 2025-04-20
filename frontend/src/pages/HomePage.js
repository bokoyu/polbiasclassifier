import React, { useState } from 'react';
import {
  Box,
  Heading,
  Textarea,
  Button,
  Text,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Tag,
  useToast
} from "@chakra-ui/react";

/**
 * Extracts label + confidence from lines like:
 *   [BIAS: Biased (0.85)]
 *   [BIAS: Neutral (0.90)]
 *   [LEANING: Left (0.76)]
 * Return format for each: { label: string, confidence: string } or null if not found
 */
function parseBiasSection(str) {
  // Matches "[BIAS: <Biased|Neutral> (<float>)]"
  const biasRegex = /\[BIAS:\s+(Neutral|Biased)\s+\(([\d.]+)\)\]/;
  const match = str.match(biasRegex);
  if (!match) return null;

  // match[1] = "Neutral" or "Biased"
  // match[2] = "0.85" or "0.90", etc.
  return {
    label: match[1],
    confidence: match[2],
  };
}

function parseLeaningSection(str) {
  // Matches "[LEANING: <Left|Right|Center> (<float>)]"
  const leanRegex = /\[LEANING:\s+(Left|Right|Center)\s+\(([\d.]+)\)\]/;
  const match = str.match(leanRegex);
  if (!match) return null;

  // match[1] = "Left" or "Right" or "Center"
  // match[2] = "0.76" or similar
  return {
    label: match[1],
    confidence: match[2],
  };
}

/**
 * Return an object with the extracted bias + leaning.
 * e.g. { biasLabel: "Biased", biasConf: "0.85", leaningLabel: "Left", leaningConf: "0.76" }
 */
function parsePrediction(rawStr) {
  const bias = parseBiasSection(rawStr);       // { label, confidence } or null
  const leaning = parseLeaningSection(rawStr); // { label, confidence } or null

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

  const handlePredict = async () => {
    setError(null);
    setPredictionRaw(null);

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
    }
  };

  const { biasLabel, biasConf, leaningLabel, leaningConf } = parsePrediction(predictionRaw || "");

  let biasColor = "blue";
  if (biasLabel === "Biased") {
    biasColor = "yellow";
  }

  let leanColor = "gray";
  if (leaningLabel === "Left") leanColor = "blue";
  if (leaningLabel === "Right") leanColor = "red";

  return (
    <Box maxW="600px" mx="auto" mt="5rem" p={4} bg="white" boxShadow="md" borderRadius="md">
      <Heading as="h1" size="lg" mb={4} textAlign="center">
        Political Bias Classifier
      </Heading>

      <Textarea
        placeholder="Enter text to classify..."
        size="md"
        resize="vertical"
        mb={4}
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
      />

      <Button colorScheme="blue" onClick={handlePredict} width="100%">
        Predict
      </Button>

      {error && (
        <Alert status="error" mt={4}>
          <AlertIcon />
          <AlertTitle>Error:</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {predictionRaw && (
        <Box mt={4}>
          <Text fontWeight="bold" mb={1}>Results:</Text>

          {biasLabel && biasConf && (
            <Tag
              size="lg"
              borderRadius="full"
              colorScheme={biasColor}
              mr={2}
              mb={2}
            >
              {`${biasLabel} (${biasConf})`}
            </Tag>
          )}

          {leaningLabel && leaningConf && (
            <Tag
              size="lg"
              borderRadius="full"
              colorScheme={leanColor}
            >
              {`${leaningLabel} (${leaningConf})`}
            </Tag>
          )}
        </Box>
      )}
    </Box>
  );
}
