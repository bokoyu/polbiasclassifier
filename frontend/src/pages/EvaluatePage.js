import React, { useState } from "react";
import {
  Heading,
  Button,
  Switch,
  FormControl,
  FormLabel,
  Box,
  useToast,
  HStack,
} from "@chakra-ui/react";

import PageCard from "../components/PageCard";
import MetricBlock from "../components/MetricBlock";
import ErrorAlert from "../components/ErrorAlert";

export default function EvaluatePage() {
  const toast = useToast();
  const [cleaning, setCleaning] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleEvaluate = async () => {
    setError(null);
    setMetrics(null);
    setIsLoading(true);

    try {
      const res = await fetch("/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cleaning }),
      });

      if (!res.ok) {
        const msg = await res.json().catch(() => ({}));
        throw new Error(msg.error || `Request failed with ${res.status}`);
      }

      const data = await res.json();
      setMetrics(data);
      toast({ title: "Evaluation complete", status: "success" });
    } catch (err) {
      setError(err.message);
      toast({ title: "Error", description: err.message, status: "error" });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <PageCard>
      <Heading size="md" mb={6} textAlign="center">
        Evaluate Models
      </Heading>

      <FormControl display="flex" alignItems="center" mb={4} justifyContent="center">
        <FormLabel mb={0}>Cleaning</FormLabel>
        <Switch
          isChecked={cleaning}
          onChange={(e) => setCleaning(e.target.checked)}
          colorScheme="blue"
        />
      </FormControl>

      <Button
        colorScheme="blue"
        onClick={handleEvaluate}
        isLoading={isLoading}
        mx="auto"
        display="block"
      >
        Evaluate
      </Button>

      {error && <ErrorAlert message={error} />}

      {metrics && (
        <HStack mt={6} spacing={6} justify="center" align="flex-start">
          <MetricBlock title="Bias" metrics={metrics.bias_metrics} />
          <MetricBlock title="Leaning" metrics={metrics.leaning_metrics} />
        </HStack>
      )}
    </PageCard>
  );
}
