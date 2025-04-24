import React, { useState, useRef, useEffect } from "react";
import {
  Heading,
  NumberInput,
  NumberInputField,
  Button,
  Switch,
  FormControl,
  FormLabel,
  useToast,
  Progress,
  Text,
} from "@chakra-ui/react";

import PageCard from "../components/PageCard";
import ErrorAlert from "../components/ErrorAlert";

export default function TrainPage() {
  const toast = useToast();

  // ─── form state ──────────────────────────────────────────────
  const [epochs, setEpochs] = useState(3);
  const [batchSize, setBatchSize] = useState(8);
  const [lrLean, setLrLean] = useState(3e-5);
  const [lrBias, setLrBias] = useState(3e-5);
  const [cleaning, setCleaning] = useState(false);

  // ─── ui state ────────────────────────────────────────────────
  const [message, setMessage] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef(null);

  const fmt = (secs) => new Date(secs * 1000).toISOString().substring(14, 19);

  const handleTrain = async () => {
    setMessage(null);
    setError(null);
    setLoading(true);
    setElapsed(0);

    timerRef.current && clearInterval(timerRef.current);
    timerRef.current = setInterval(() => setElapsed((s) => s + 1), 1000);

    try {
      const res = await fetch("/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          epochs: Number(epochs),
          batch_size: Number(batchSize),
          lr_bias: Number(lrBias),
          lr_lean: Number(lrLean),
          cleaning,
        }),
      });

      if (!res.ok) {
        const msg = await res.json().catch(() => ({}));
        throw new Error(msg.error || `Request failed with ${res.status}`);
      }

      const data = await res.json();
      setMessage(data.message);
      toast({ title: "Training started", description: data.message, status: "success" });
    } catch (err) {
      setError(err.message);
      toast({ title: "Error", description: err.message, status: "error" });
    } finally {
      timerRef.current && clearInterval(timerRef.current);
      timerRef.current = null;
      setLoading(false);
    }
  };

  useEffect(() => () => timerRef.current && clearInterval(timerRef.current), []);

  return (
    <PageCard>
      <Heading size="md" mb={6} textAlign="center">
        Train Models
      </Heading>

      <FormControl mb={4}>
        <FormLabel>Epochs</FormLabel>
        <NumberInput min={1} value={epochs} onChange={(val) => setEpochs(val)}>
          <NumberInputField />
        </NumberInput>
      </FormControl>

      <FormControl mb={4}>
        <FormLabel>Batch Size</FormLabel>
        <NumberInput min={1} value={batchSize} onChange={(val) => setBatchSize(val)}>
          <NumberInputField />
        </NumberInput>
      </FormControl>

      <FormControl mb={4}>
        <FormLabel>Learning Rate (Leaning Model)</FormLabel>
        <NumberInput min={1e-8} step={1e-5} value={lrLean} onChange={(val) => setLrLean(val)}>
          <NumberInputField />
        </NumberInput>
      </FormControl>

      <FormControl mb={4}>
        <FormLabel>Learning Rate (Bias Model)</FormLabel>
        <NumberInput min={1e-8} step={1e-5} value={lrBias} onChange={(val) => setLrBias(val)}>
          <NumberInputField />
        </NumberInput>
      </FormControl>

      <FormControl display="flex" alignItems="center" mb={6} justifyContent="center">
        <FormLabel mb={0}>Cleaning</FormLabel>
        <Switch isChecked={cleaning} onChange={(e) => setCleaning(e.target.checked)} colorScheme="blue" />
      </FormControl>

      <Button colorScheme="blue" onClick={handleTrain} isLoading={loading} mx="auto" display="block">
        Train
      </Button>

      {loading && (
        <>
          <Progress mt={6} isIndeterminate size="sm" borderRadius="full" />
          <Text mt={2} fontSize="sm" color="gray.600" textAlign="center">
            Training… elapsed {fmt(elapsed)}
          </Text>
        </>
      )}

      {error && <ErrorAlert message={error} />}
      {message && !error && !loading && <ErrorAlert message={message} status="success" />}
    </PageCard>
  );
}
