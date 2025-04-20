import React, { useState } from 'react';
import {
  Box,
  Heading,
  Button,
  Input,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Text,
  Switch,
  FormControl,
  FormLabel,
  useToast
} from '@chakra-ui/react';

function EvaluatePage() {
  const toast = useToast();

  // State for user inputs
  const [dataPath, setDataPath] = useState("");
  const [cleaning, setCleaning] = useState(false);

  // State for results
  const [metrics, setMetrics] = useState(null);
  // For any fetch error or server error
  const [error, setError] = useState(null);

  // Called when user clicks "Evaluate"
  const handleEvaluate = async () => {
    setError(null);
    setMetrics(null);

    try {
      const response = await fetch('/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data_path: dataPath,
          cleaning: cleaning
        }),
      });

      if (!response.ok) {
        // Attempt to parse JSON error message
        const msg = await response.json();
        throw new Error(msg.error || 'Request failed');
      }

      const data = await response.json();
      // data => { bias_metrics: {...}, leaning_metrics: {...} }
      setMetrics(data);

      // Optionally show a success toast
      toast({
        title: "Evaluation Complete",
        description: "Successfully evaluated the models!",
        status: "success",
        duration: 4000,
        isClosable: true,
      });
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

  return (
    <Box maxW="600px" mx="auto" mt="5rem" p={4} bg="white" boxShadow="md" borderRadius="md">
      <Heading as="h2" size="lg" mb={4}>Evaluate Models</Heading>

      {/* Data Path Input */}
      <Input
        placeholder="Path to data file"
        mb={4}
        value={dataPath}
        onChange={(e) => setDataPath(e.target.value)}
      />

      {/* Cleaning Toggle */}
      <FormControl display="flex" alignItems="center" mb={4}>
        <FormLabel mb="0">Cleaning</FormLabel>
        <Switch
          isChecked={cleaning}
          onChange={(e) => setCleaning(e.target.checked)}
          colorScheme="blue"
        />
      </FormControl>

      {/* Evaluate Button */}
      <Button colorScheme="blue" onClick={handleEvaluate}>
        Evaluate
      </Button>

      {/* Error Alert */}
      {error && (
        <Alert status="error" mt={4}>
          <AlertIcon />
          <AlertTitle>Error:</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Metrics Display */}
      {metrics && (
        <Box mt={6}>
          <Heading as="h3" size="md" mb={2}>Bias Model Metrics</Heading>
          <Box mb={4}>
            <Text>Accuracy: {metrics.bias_metrics?.accuracy?.toFixed(4)}</Text>
            <Text>Precision: {metrics.bias_metrics?.precision?.toFixed(4)}</Text>
            <Text>Recall: {metrics.bias_metrics?.recall?.toFixed(4)}</Text>
            <Text>F1: {metrics.bias_metrics?.f1?.toFixed(4)}</Text>
          </Box>

          <Heading as="h3" size="md" mb={2}>Leaning Model Metrics</Heading>
          <Box>
            <Text>Accuracy: {metrics.leaning_metrics?.accuracy?.toFixed(4)}</Text>
            <Text>Precision: {metrics.leaning_metrics?.precision?.toFixed(4)}</Text>
            <Text>Recall: {metrics.leaning_metrics?.recall?.toFixed(4)}</Text>
            <Text>F1: {metrics.leaning_metrics?.f1?.toFixed(4)}</Text>
          </Box>
        </Box>
      )}
    </Box>
  );
}

export default EvaluatePage;
