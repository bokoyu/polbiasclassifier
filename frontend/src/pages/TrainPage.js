// src/pages/TrainPage.js
import React, { useState } from 'react';
import {
  Box,
  Heading,
  Input,
  NumberInput,
  NumberInputField,
  Button,
  Switch,
  FormControl,
  FormLabel,
  Text,
  useToast
} from '@chakra-ui/react';

function TrainPage() {
  const [dataPath, setDataPath] = useState("");
  const [epochs, setEpochs] = useState(3);
  const [batchSize, setBatchSize] = useState(8);
  const [lrLean, setLrLean] = useState(3e-5);
  const [lrBias, setLrBias] = useState(3e-5);
  const [cleaning, setCleaning] = useState(false);
  const [message, setMessage] = useState(null);

  const toast = useToast();

  const handleTrain = async () => {
    setMessage(null);

    try {
      const response = await fetch('/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data_path: dataPath,
          epochs: parseInt(epochs),
          batch_size: parseInt(batchSize),
          lr_bias: parseFloat(lrBias),
          lr_lean: parseFloat(lrLean),
          cleaning,
        }),
      });
      if (!response.ok) {
        const msg = await response.json();
        throw new Error(msg.error || 'Request failed');
      }
      const data = await response.json();
      setMessage(data.message);
    } catch (err) {
      toast({
        title: 'Error',
        description: err.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  return (
    <Box
      maxW="600px"
      mx="auto"
      mt="5rem"
      p={4}
      bg="white"
      borderRadius="md"
      boxShadow="md"
    >
      <Heading as="h2" size="lg" mb={4}>Train Models</Heading>

      <Input
        placeholder="Path to data file"
        mb={4}
        value={dataPath}
        onChange={(e) => setDataPath(e.target.value)}
      />

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
      {/* LEANING LR */}
      <FormControl mb={4}>
        <FormLabel>Learning Rate (Leaning Model)</FormLabel>
        <NumberInput
          min={1e-8}
          step={1e-5}
          value={lrLean}
          onChange={(val) => setLrLean(val)}
        >
          <NumberInputField />
        </NumberInput>
      </FormControl>

      {/* BIAS LR */}
      <FormControl mb={4}>
        <FormLabel>Learning Rate (Bias Model)</FormLabel>
        <NumberInput
          min={1e-8}
          step={1e-5}
          value={lrBias}
          onChange={(val) => setLrBias(val)}
        >
          <NumberInputField />
        </NumberInput>
      </FormControl>

      <FormControl display="flex" alignItems="center" mb={4}>
        <FormLabel mb="0">Cleaning</FormLabel>
        <Switch
          isChecked={cleaning}
          onChange={(e) => setCleaning(e.target.checked)}
          colorScheme="blue"
        />
      </FormControl>

      <Button colorScheme="blue" onClick={handleTrain}>
        Train
      </Button>

      {message && (
        <Text color="green.600" mt={4}>
          {message}
        </Text>
      )}
    </Box>
  );
}

export default TrainPage;
