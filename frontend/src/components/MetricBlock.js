import { Box, Text, VStack } from "@chakra-ui/react";
export default function MetricBlock({ title, metrics = {} }) {
  return (
    <Box>
      <Text fontWeight="semibold" mb={1}>{title}</Text>
      <VStack spacing={0} align="flex-start" fontSize="sm">
        {Object.entries(metrics).map(([k, v]) => (
          <Text key={k}>{`${k}: ${v.toFixed ? v.toFixed(4) : v}`}</Text>
        ))}
      </VStack>
    </Box>
  );
}